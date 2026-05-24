import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

# ==========================================
# 0. 初始化 Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("Signal_Engine_Trip_Splitter_T_Plus_1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# ==========================================
# 1. 定义 UDF 输出 Schema
# ==========================================
# 扩充了状态字段以支持严格的状态机回溯和断网兜底
result_schema = StructType([
    StructField("vin", StringType()),
    StructField("row_type", StringType()),      # 'TRIP': 产出的行程; 'STATE': 日终状态
    StructField("trip_type", StringType()),     # 'DRIVING' / 'PARKING'
    StructField("phase", StringType()),         # 内部引擎状态: 'OFF' / 'ACTIVE' / 'HANG_OFF'
    StructField("start_time", TimestampType()), 
    StructField("end_time", TimestampType()),   
    StructField("max_speed", DoubleType()),     
    StructField("avg_speed", DoubleType()),     
    StructField("last_ts", TimestampType()),    # 最后活跃心跳 (用于15分钟断网判定)
    StructField("last_speed_ts", TimestampType()), # 最后一次速度>0的时间 (用于长时怠速切分回溯)
    StructField("state_speeds", StringType())   # 未结算行程的速度序列透传
])

# ==========================================
# 2. 核心状态机逻辑 (Pandas UDF)
# ==========================================
def process_t_plus_1_with_metrics(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
    vin = key[0]
    output_rows = []
    
    # 消除时序跳变：昨日状态在前，今日日志按时间严格排序
    pdf = pdf.sort_values(by=['timestamp', 'is_state'], ascending=[True, False])
    
    # 初始化状态机变量
    phase = 'OFF'
    trip_type_so_far = 'PARKING'
    trip_start = None
    hang_start = None
    last_ts = None
    last_speed_ts = None
    trip_speeds = [] 
    
    # 辅助函数：生成并过滤有效行程记录
    def emit_trip(t_type, t_start, t_end, speeds):
        if pd.isna(t_start) or pd.isna(t_end) or t_start > t_end:
            return None
        max_s = float(np.max(speeds)) if speeds else 0.0
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        return (vin, 'TRIP', t_type, None, t_start, t_end, max_s, avg_s, t_end, None, None)

    # 遍历事件流，驱动状态机
    for _, row in pdf.iterrows():
        # --- 恢复 T-1 的状态 ---
        if row['is_state'] == 1:
            phase = row['state_phase'] if pd.notna(row['state_phase']) else 'OFF'
            trip_type_so_far = row['state_trip_type'] if pd.notna(row['state_trip_type']) else 'PARKING'
            trip_start = row['state_trip_start'] if pd.notna(row['state_trip_start']) else None
            hang_start = row['state_hang_start'] if pd.notna(row['state_hang_start']) else None
            last_ts = row['timestamp'] if pd.notna(row['timestamp']) else None
            last_speed_ts = row['state_last_speed_ts'] if pd.notna(row['state_last_speed_ts']) else None
            
            speeds_str = row.get('state_speeds', '')
            if pd.notna(speeds_str) and str(speeds_str).strip() != '':
                trip_speeds = [float(x) for x in str(speeds_str).split(',')]
            continue
            
        # --- 处理 T 日日志 ---
        ts = row['timestamp']
        power = row['syspowermod_2012001']
        speed = float(row['vehspd_2011002']) if pd.notna(row['vehspd_2011002']) else 0.0
        
        # [规则 4] 异常兜底 (防死锁)：检测数据断层是否超过 15 分钟
        if phase != 'OFF' and last_ts is not None:
            if (ts - last_ts).total_seconds() > 900:
                # 强制结算当前所有挂起或进行中的行程，结束时间回溯到 last_ts 或 hang_start
                end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                if trip_record: output_rows.append(trip_record)
                
                # 重置引擎状态
                phase, trip_speeds = 'OFF', []
                trip_type_so_far, last_speed_ts = 'PARKING', None

        # 状态机流转评估
        if phase == 'OFF':
            # [规则 1] 行程开始 (信号补偿)
            if power == 2 or speed > 0:
                phase = 'ACTIVE'
                trip_start = ts
                trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                last_speed_ts = ts if speed > 0 else None
                trip_speeds = [speed]

        elif phase == 'HANG_OFF':
            # [规则 3] 下电合并与结束 (5分钟防抖)
            if (ts - hang_start).total_seconds() > 300:
                # 满 5 分钟未恢复，正式结算上一段行程
                trip_record = emit_trip(trip_type_so_far, trip_start, hang_start, trip_speeds)
                if trip_record: output_rows.append(trip_record)
                
                phase, trip_speeds = 'OFF', []
                trip_type_so_far, last_speed_ts = 'PARKING', None
                
                # 重新评估当前行是否触发新行程
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_start = ts
                    trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                    last_speed_ts = ts if speed > 0 else None
                    trip_speeds = [speed]
            else:
                # 5 分钟内重新上电或有速度，恢复行程并合并
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_speeds.append(speed)
                    if speed > 0:
                        trip_type_so_far = 'DRIVING'
                        last_speed_ts = ts

        if phase == 'ACTIVE':
            split_happened = False
            
            # [规则 2] 长时怠速切分 (15分钟连续速度为0)
            if trip_type_so_far == 'DRIVING' and speed == 0 and last_speed_ts is not None:
                if (ts - last_speed_ts).total_seconds() > 900:
                    # 剥离前半段行驶，结束时间精确回溯至最后一次有速度的时刻
                    trip_record = emit_trip('DRIVING', trip_start, last_speed_ts, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    
                    # 无缝衔接开启新的驻车行程
                    trip_start = last_speed_ts
                    trip_type_so_far = 'PARKING'
                    last_speed_ts = None
                    trip_speeds = [0.0]
                    split_happened = True
                    
            elif trip_type_so_far == 'PARKING' and speed > 0:
                if (ts - trip_start).total_seconds() > 900:
                    # 纯驻车行程被终结，结算之前的纯驻车数据
                    trip_record = emit_trip('PARKING', trip_start, last_ts, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    
                    # 立刻开启新的行驶行程
                    trip_start = last_ts
                    trip_type_so_far = 'DRIVING'
                    last_speed_ts = ts
                    trip_speeds = [speed]
                    split_happened = True
                else:
                    # 短时驻车，直接晋升合并为行驶行程
                    trip_type_so_far = 'DRIVING'
                    last_speed_ts = ts

            # 常规日志追加
            if not split_happened:
                trip_speeds.append(speed)
                if speed > 0:
                    last_speed_ts = ts

            # 评估是否触发下电挂起
            if power == 0:
                phase = 'HANG_OFF'
                hang_start = ts

        # 更新系统水位线
        last_ts = ts
                    
    # --- 批次处理结束，生成供明天使用的日终状态快照 ---
    state_speeds_str = ','.join(map(str, trip_speeds)) if phase != 'OFF' and trip_speeds else None
    output_rows.append((vin, 'STATE', trip_type_so_far, phase, trip_start, hang_start, None, None, last_ts, last_speed_ts, state_speeds_str))

    return pd.DataFrame(output_rows, columns=result_schema.names)

# ==========================================
# 3. 数据准备与 Pipeline 执行
# ==========================================
def run_t_plus_1_job(target_date: str, prev_date: str):
    try:
        state_df = spark.table("dwd_veh_trip_state").filter(F.col("dt") == prev_date)
    except Exception:
        state_df = spark.createDataFrame([], schema=StructType([]))

    # 字段对齐，注意加入了 state_trip_type 和 state_last_speed_ts
    state_aligned = state_df.select(
        F.col("vin"),
        F.col("last_ts").alias("timestamp"),
        F.lit(None).cast(IntegerType()).alias("syspowermod_2012001"),
        F.lit(None).cast(DoubleType()).alias("vehspd_2011002"),
        F.lit(1).cast(IntegerType()).alias("is_state"),
        F.col("phase").alias("state_phase"),
        F.col("trip_type").alias("state_trip_type"),
        F.col("trip_start").alias("state_trip_start"),
        F.col("hang_start").alias("state_hang_start"),
        F.col("last_speed_ts").alias("state_last_speed_ts"),
        F.col("state_speeds")
    )

    raw_df = spark.table("ods_veh_log").filter(F.col("dt") == target_date)

    raw_aligned = raw_df.select(
        F.col("vin"),
        F.col("timestamp"),
        F.col("syspowermod_2012001"),
        F.col("vehspd_2011002"),
        F.lit(0).cast(IntegerType()).alias("is_state"),
        F.lit(None).cast(StringType()).alias("state_phase"),
        F.lit(None).cast(StringType()).alias("state_trip_type"),
        F.lit(None).cast(TimestampType()).alias("state_trip_start"),
        F.lit(None).cast(TimestampType()).alias("state_hang_start"),
        F.lit(None).cast(TimestampType()).alias("state_last_speed_ts"),
        F.lit(None).cast(StringType()).alias("state_speeds")
    )

    union_df = state_aligned.unionByName(raw_aligned)
    
    processed_df = union_df.groupBy("vin").applyInPandas(
        process_t_plus_1_with_metrics, 
        schema=result_schema
    )
    
    processed_df.cache()

    # ==========================================
    # 4. 数据分流落表
    # ==========================================
    
    # A. 写入行程结果表 (dwd_veh_trips_table)
    trips_output = processed_df.filter(F.col("row_type") == 'TRIP').select(
        "vin", 
        "trip_type", 
        "start_time", 
        "end_time",
        "max_speed",
        "avg_speed"
    ).withColumn("dt", F.lit(target_date))
    
    trips_output.write.mode("append").partitionBy("dt").insertInto("dwd_veh_trips_table")

    # B. 覆写 T 日状态表快照 (dwd_veh_trip_state)
    new_state_output = processed_df.filter(F.col("row_type") == 'STATE').select(
        "vin",
        "phase",
        "trip_type",
        "start_time",
        F.col("end_time").alias("hang_start"),
        "last_ts",
        "last_speed_ts",
        "state_speeds"
    ).withColumn("dt", F.lit(target_date))

    new_state_output.write.mode("overwrite").partitionBy("dt").insertInto("dwd_veh_trip_state")

    processed_df.unpersist()