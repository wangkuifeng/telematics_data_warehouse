import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import DataFrame

# ==========================================
# 1. 定义 UDF 输出 Schema
# ==========================================
RESULT_SCHEMA = StructType([
    StructField("vin", StringType()),
    StructField("row_type", StringType()),      
    StructField("trip_type", StringType()),     
    StructField("phase", StringType()),         
    StructField("start_time", TimestampType()), 
    StructField("end_time", TimestampType()),   
    StructField("max_speed", DoubleType()),     
    StructField("avg_speed", DoubleType()),     
    StructField("last_ts", TimestampType()),    
    StructField("last_speed_ts", TimestampType()), 
    StructField("state_speeds", StringType())   
])

# ==========================================
# 2. 核心状态机逻辑 (Pandas UDF)
# ==========================================
def process_t_plus_1_with_metrics(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
    vin = key[0]
    output_rows = []
    
    pdf = pdf.sort_values(by=['timestamp', 'is_state'], ascending=[True, False])
    
    phase = 'OFF'
    trip_type_so_far = 'PARKING'
    trip_start = None
    hang_start = None
    last_ts = None
    last_speed_ts = None
    trip_speeds = [] 
    
    def emit_trip(t_type, t_start, t_end, speeds):
        if pd.isna(t_start) or pd.isna(t_end) or t_start > t_end:
            return None
        max_s = float(np.max(speeds)) if speeds else 0.0
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        return (vin, 'TRIP', t_type, None, t_start, t_end, max_s, avg_s, t_end, None, None)

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
        
        # [规则 4] 异常兜底 (防死锁)
        if phase != 'OFF' and last_ts is not None:
            if (ts - last_ts).total_seconds() > 900:
                end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                if trip_record: output_rows.append(trip_record)
                
                phase, trip_speeds = 'OFF', []
                trip_type_so_far, last_speed_ts = 'PARKING', None

        # 状态机流转评估
        if phase == 'OFF':
            if power == 2 or speed > 0:
                phase = 'ACTIVE'
                trip_start = ts
                trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                last_speed_ts = ts if speed > 0 else None
                trip_speeds = [speed]

        elif phase == 'HANG_OFF':
            if (ts - hang_start).total_seconds() > 300:
                trip_record = emit_trip(trip_type_so_far, trip_start, hang_start, trip_speeds)
                if trip_record: output_rows.append(trip_record)
                
                phase, trip_speeds = 'OFF', []
                trip_type_so_far, last_speed_ts = 'PARKING', None
                
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_start = ts
                    trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                    last_speed_ts = ts if speed > 0 else None
                    trip_speeds = [speed]
            else:
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_speeds.append(speed)
                    if speed > 0:
                        trip_type_so_far = 'DRIVING'
                        last_speed_ts = ts

        if phase == 'ACTIVE':
            split_happened = False
            
            if trip_type_so_far == 'DRIVING' and speed == 0 and last_speed_ts is not None:
                if (ts - last_speed_ts).total_seconds() > 900:
                    trip_record = emit_trip('DRIVING', trip_start, last_speed_ts, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    
                    trip_start = last_speed_ts
                    trip_type_so_far = 'PARKING'
                    last_speed_ts = None
                    trip_speeds = [0.0]
                    split_happened = True
                    
            elif trip_type_so_far == 'PARKING' and speed > 0:
                if (ts - trip_start).total_seconds() > 900:
                    trip_record = emit_trip('PARKING', trip_start, last_ts, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    
                    trip_start = last_ts
                    trip_type_so_far = 'DRIVING'
                    last_speed_ts = ts
                    trip_speeds = [speed]
                    split_happened = True
                else:
                    trip_type_so_far = 'DRIVING'
                    last_speed_ts = ts

            if not split_happened:
                trip_speeds.append(speed)
                if speed > 0:
                    last_speed_ts = ts

            if power == 0:
                phase = 'HANG_OFF'
                hang_start = ts

        last_ts = ts
                    
    # --- 批次处理结束，生成供明天使用的日终状态快照 ---
    state_speeds_str = ','.join(map(str, trip_speeds)) if phase != 'OFF' and trip_speeds else None
    output_rows.append((vin, 'STATE', trip_type_so_far, phase, trip_start, hang_start, None, None, last_ts, last_speed_ts, state_speeds_str))

    return pd.DataFrame(output_rows, columns=RESULT_SCHEMA.names)

# ==========================================
# 3. 核心转换 Pipeline
# ==========================================
def transform_trip_data(state_df: DataFrame, raw_df: DataFrame) -> DataFrame:
    """
    接收原始的昨日状态表和今日日志表，执行清洗、对齐与状态机计算。
    返回包含 TRIP 和 STATE 的聚合 DataFrame。
    """
    # 字段对齐
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
    
    # 执行计算
    processed_df = union_df.groupBy("vin").applyInPandas(
        process_t_plus_1_with_metrics, 
        schema=RESULT_SCHEMA
    )
    
    return processed_df