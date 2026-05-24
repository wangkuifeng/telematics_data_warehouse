import pandas as pd
import numpy as np
import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

# ==========================================
# 0. 初始化 Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("Signal_Engine_Trip_Logic_Full_Test") \
    .master("local[*]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 定义 UDF 输出 Schema
result_schema = StructType([
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
# 1. 核心状态机 UDF (严格对齐文档)
# ==========================================
def process_trip_state_machine(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
    vin = key[0]
    output_rows = []
    
    # 时序排序
    pdf = pdf.sort_values(by=['timestamp', 'is_state'], ascending=[True, False])
    
    # 状态机初始化
    phase = 'OFF'
    trip_type_so_far = 'PARKING'
    trip_start = None
    hang_start = None
    last_ts = None
    last_speed_ts = None
    trip_speeds = [] 
    
    def emit_trip(t_type, t_start, t_end, speeds):
        if pd.isna(t_start) or pd.isna(t_end) or t_start >= t_end:
            return None
        max_s = float(np.max(speeds)) if speeds else 0.0
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        return (vin, 'TRIP', t_type, None, t_start, t_end, max_s, avg_s, t_end, None, None)

    for _, row in pdf.iterrows():
        # 简化版：仅处理当日日志
        ts = row['timestamp']
        power = row['syspowermod_2012001']
        speed = float(row['vehspd_2011002']) if pd.notna(row['vehspd_2011002']) else 0.0
        
        # [规则 4] 异常兜底 (防死锁)：检测数据断层超 15 分钟
        if phase != 'OFF' and last_ts is not None:
            if (ts - last_ts).total_seconds() > 900:
                end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                if trip_record: output_rows.append(trip_record)
                phase, trip_speeds = 'OFF', []
                trip_type_so_far, last_speed_ts = 'PARKING', None

        # 状态跃迁逻辑
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
                    
    # 批次结束清理残存状态 (用于单独跑批验证)
    if phase != 'OFF':
        end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
        trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
        if trip_record: output_rows.append(trip_record)

    return pd.DataFrame(output_rows, columns=result_schema.names)

# ==========================================
# 2. 真实车辆行为数据生成器 (100辆车, 1分钟/条)
# ==========================================
print("\n" + "🚀"*3 + " 开始生成 100 辆车全场景路测数据 (采样率: 1 分钟/条) " + "🚀"*3)

def append_action(data_list, vin, start_time, duration_minutes, power, speed, label=""):
    current_time = start_time
    for _ in range(int(duration_minutes)):
        actual_speed = round(speed + np.random.uniform(-2, 2), 2) if speed > 0 else 0.0
        data_list.append((vin, current_time, power, actual_speed, label))
        current_time += datetime.timedelta(minutes=1)
    return current_time

raw_data = []
base_time = datetime.datetime(2026, 5, 20, 8, 0, 0)

# 分配 100 辆车到 6 大场景
scenario_distribution = {
    'A': 20, # 场景A: 正常通勤 (20辆)
    'B': 10, # 场景B: 短暂抽烟纯驻车 (10辆)
    'C': 20, # 场景C: 中途长时等待 (20辆)
    'D': 20, # 场景D: 加油站短时下电合并 (20辆)
    'E': 10, # 场景E: 进地库断网 (10辆)
    'F': 20  # 场景F: 自动启停/极端短下电 (20辆)
}

vin_counter = 1
for scenario, count in scenario_distribution.items():
    for _ in range(count):
        vin = f"LSV_SCENARIO_{scenario}_{vin_counter:03d}"
        t = base_time
        
        if scenario == 'A':
            # 正常通勤：开10分 -> 红绿灯3分 -> 开10分 -> 熄火
            t = append_action(raw_data, vin, t, 10, 2, 45.0, "行驶")
            t = append_action(raw_data, vin, t, 3, 2, 0.0, "等红绿灯")
            t = append_action(raw_data, vin, t, 10, 2, 40.0, "行驶")
            raw_data.append((vin, t, 0, 0.0, "熄火下电"))
            
        elif scenario == 'B':
            # 纯驻车：上电不走5分钟 -> 熄火
            t = append_action(raw_data, vin, t, 5, 2, 0.0, "车内休息")
            raw_data.append((vin, t, 0, 0.0, "熄火下电"))
            
        elif scenario == 'C':
            # 中途长时等待切分：开10分 -> 路边等人20分 -> 开10分 -> 熄火
            t = append_action(raw_data, vin, t, 10, 2, 35.0, "行驶")
            t = append_action(raw_data, vin, t, 20, 2, 0.0, "长时间等人(触发15分切分)")
            t = append_action(raw_data, vin, t, 10, 2, 50.0, "行驶")
            raw_data.append((vin, t, 0, 0.0, "熄火下电"))
            
        elif scenario == 'D':
            # 加油站短时下电：开20分 -> 熄火3分 -> 重新点火开10分 -> 熄火
            t = append_action(raw_data, vin, t, 20, 2, 60.0, "行驶")
            raw_data.append((vin, t, 0, 0.0, "进站熄火"))
            t += datetime.timedelta(minutes=3) # 熄火3分钟
            t = append_action(raw_data, vin, t, 10, 2, 55.0, "继续行驶")
            raw_data.append((vin, t, 0, 0.0, "最终熄火"))
            
        elif scenario == 'E':
            # 进地库断网：开15分 -> 停车1分 -> 彻底丢失信号
            t = append_action(raw_data, vin, t, 15, 2, 30.0, "行驶")
            t = append_action(raw_data, vin, t, 1, 2, 0.0, "进地库找车位")
            # 模拟2小时后才重连(强制触发15分钟Gap Timeout)
            t += datetime.timedelta(hours=2)
            raw_data.append((vin, t, 2, 0.0, "次日重新上电(触发昨日兜底结算)"))

        elif scenario == 'F':
            # 自动启停短下电：开10分 -> 极短熄火1分 -> 开10分 -> 熄火
            t = append_action(raw_data, vin, t, 10, 2, 35.0, "行驶")
            raw_data.append((vin, t, 0, 0.0, "启停下电"))
            t += datetime.timedelta(minutes=1) # 1分钟恢复
            t = append_action(raw_data, vin, t, 10, 2, 40.0, "继续行驶")
            raw_data.append((vin, t, 0, 0.0, "最终熄火"))

        vin_counter += 1

# 转换为 DataFrame
raw_df = spark.createDataFrame(raw_data, schema=["vin", "timestamp", "syspowermod_2012001", "vehspd_2011002", "action_label"])

input_df = raw_df.select(
    F.col("vin"), F.col("timestamp"), F.col("syspowermod_2012001"), F.col("vehspd_2011002"),
    F.lit(0).cast(IntegerType()).alias("is_state")
)

# ==========================================
# 3. 执行状态机切分并输出核心指标
# ==========================================
print("\n" + "🔥"*3 + " 开始执行引擎跑批 (分布式 Pandas UDF) " + "🔥"*3)

processed_df = input_df.groupBy("vin").applyInPandas(process_trip_state_machine, schema=result_schema).cache()

output_df = processed_df.filter(F.col("row_type") == 'TRIP').select(
    "vin",
    "trip_type",
    F.date_format("start_time", "HH:mm:ss").alias("开始时间"), 
    F.date_format("end_time", "HH:mm:ss").alias("结束时间"),
    F.round("max_speed", 2).alias("Max_Speed"),
    F.round("avg_speed", 2).alias("Avg_Speed"),
    F.round((F.unix_timestamp("end_time") - F.unix_timestamp("start_time")) / 60, 1).alias("时长(分钟)")
).orderBy("vin", "开始时间")

# 抽取每个场景的代表车辆进行打印展示
print("\n=== 场景 A 代表车 (红绿灯防切碎验证) ===")
output_df.filter(F.col("vin") == "LSV_SCENARIO_A_001").show(truncate=False)

print("\n=== 场景 B 代表车 (纯车内休息验证) ===")
output_df.filter(F.col("vin") == "LSV_SCENARIO_B_021").show(truncate=False)

print("\n=== 场景 C 代表车 (中途超15分钟长怠速硬切分验证) ===")
output_df.filter(F.col("vin") == "LSV_SCENARIO_C_031").show(truncate=False)

print("\n=== 场景 D 代表车 (加油站 5分钟内下电防抖合并验证) ===")
output_df.filter(F.col("vin") == "LSV_SCENARIO_D_051").show(truncate=False)

print("\n=== 场景 E 代表车 (进地库直接断网/掉电 兜底结算验证) ===")
output_df.filter(F.col("vin") == "LSV_SCENARIO_E_071").show(truncate=False)

print("\n=== 场景 F 代表车 (自动启停极短下电 合并验证) ===")
output_df.filter(F.col("vin") == "LSV_SCENARIO_F_081").show(truncate=False)

print("\n📊 总体数据产出统计:")
output_df.groupBy("trip_type").count().show()

spark.stop()