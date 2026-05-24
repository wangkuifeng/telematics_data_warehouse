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
    .appName("V2X_Trip_CoreLogic_Test") \
    .master("local[*]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

result_schema = StructType([
    StructField("vin", StringType()),
    StructField("row_type", StringType()),      
    StructField("trip_type", StringType()),     
    StructField("start_time", TimestampType()), 
    StructField("end_time", TimestampType()),   
    StructField("max_speed", DoubleType()),     
    StructField("avg_speed", DoubleType()),     
    StructField("last_ts", TimestampType()),    
    StructField("state_speeds", StringType())   
])

# ==========================================
# 1. 核心 UDF 逻辑 (无需修改)
# ==========================================
def process_t_plus_1_with_metrics(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
    vin = key[0]
    output_rows = []
    
    pdf = pdf.sort_values(by=['timestamp', 'is_state'], ascending=[True, False])
    
    phase, trip_start, stop_start = 'OFF', None, None
    last_ts = None
    trip_speeds = [] 
    
    SPEED_THRESHOLD = 3.0
    MAX_GAP_SECONDS = 12 * 3600 
    
    def emit_trip(trip_t, t_start, t_end, speeds):
        if not speeds:
            return (vin, 'TRIP', trip_t, t_start, t_end, 0.0, 0.0, t_end, None)
        return (vin, 'TRIP', trip_t, t_start, t_end, float(np.max(speeds)), float(np.mean(speeds)), t_end, None)

    for _, row in pdf.iterrows():
        if row['is_state'] == 1:
            phase = row['state_phase']
            trip_start = row['state_trip_start']
            stop_start = row['state_stop_start']
            last_ts = row['timestamp']
            state_speeds_str = row.get('state_speeds', '')
            trip_speeds = [float(x) for x in state_speeds_str.split(',')] if pd.notna(state_speeds_str) and state_speeds_str else []
            continue
            
        ts = row['timestamp']
        power = row['syspowermod_2012001']
        speed = row['vehspd_2011002']
        
        if last_ts and (ts - last_ts).total_seconds() > MAX_GAP_SECONDS:
            if phase != 'OFF':
                if phase == 'DRIVE' or phase == 'STOP_WAIT':
                    output_rows.append(emit_trip('DRIVING', trip_start, last_ts, trip_speeds))
                phase, trip_start, stop_start, trip_speeds = 'OFF', None, None, []
                
        last_ts = ts 
        
        if phase == 'OFF':
            if power == 2 and speed <= SPEED_THRESHOLD:
                phase, trip_start, trip_speeds = 'PRE_PARK', ts, []
            elif power == 2 and speed > SPEED_THRESHOLD:
                phase, trip_start, trip_speeds = 'DRIVE', ts, [speed]

        elif phase == 'PRE_PARK':
            if speed > SPEED_THRESHOLD:
                duration_mins = (ts - trip_start).total_seconds() / 60.0
                if duration_mins > 15:
                    output_rows.append(emit_trip('PRE_PARKING', trip_start, ts, []))
                    trip_start, trip_speeds = ts, [speed]
                else:
                    trip_speeds.append(speed)
                phase = 'DRIVE'
            elif power == 0:
                phase = 'OFF'

        elif phase == 'DRIVE':
            if speed > SPEED_THRESHOLD:
                trip_speeds.append(speed)
            if speed <= SPEED_THRESHOLD or power == 0:
                phase, stop_start = 'STOP_WAIT', ts

        elif phase == 'STOP_WAIT':
            if speed > SPEED_THRESHOLD:
                trip_speeds.append(speed)
            duration_mins = (ts - stop_start).total_seconds() / 60.0
            
            if speed > SPEED_THRESHOLD and power == 2:
                if duration_mins <= 15:
                    phase = 'DRIVE'
                else:
                    drive_speeds = trip_speeds[:-1] 
                    output_rows.append(emit_trip('DRIVING', trip_start, stop_start, drive_speeds))
                    output_rows.append(emit_trip('POST_PARKING', stop_start, ts, [0.0]))
                    phase, trip_start, trip_speeds = 'DRIVE', ts, [speed]
            elif power == 0:
                if duration_mins <= 15:
                    output_rows.append(emit_trip('DRIVING', trip_start, ts, trip_speeds))
                    phase = 'OFF'
                else:
                    drive_speeds = trip_speeds[:-1]
                    output_rows.append(emit_trip('DRIVING', trip_start, stop_start, drive_speeds))
                    output_rows.append(emit_trip('POST_PARKING', stop_start, ts, [0.0]))
                    phase = 'OFF'
                    
    state_speeds_str = ','.join(map(str, trip_speeds)) if phase != 'OFF' and trip_speeds else None
    output_rows.append((vin, 'STATE', phase, trip_start, stop_start, None, None, last_ts, state_speeds_str))

    return pd.DataFrame(output_rows, columns=result_schema.names)

# ==========================================
# 2. 构造 1秒/条 密度的动作模拟数据
# ==========================================
def append_action(data_list, vin, start_time, duration_minutes, power, speed, label=""):
    """辅助函数：模拟 1秒1条 的连续数据，方便计算 avg_speed"""
    current_time = start_time
    for _ in range(int(duration_minutes * 60)):
        # 加上一点随机抖动让速度更真实
        actual_speed = speed + np.random.uniform(-2, 2) if speed > 0 else 0.0
        data_list.append((vin, current_time, power, actual_speed, label))
        current_time += datetime.timedelta(seconds=1)
    return current_time

def generate_core_logic_data():
    data = []
    
    # 🚗 [测试 1] 行驶合并测试 (红绿灯 <= 15分)
    t = datetime.datetime(2023, 10, 2, 8, 0, 0)
    t = append_action(data, "VIN_1_DRIVE", t, duration_minutes=10, power=2, speed=40.0, label="起步行驶")
    t = append_action(data, "VIN_1_DRIVE", t, duration_minutes=5, power=2, speed=0.0, label="等红绿灯(5分)")
    t = append_action(data, "VIN_1_DRIVE", t, duration_minutes=10, power=2, speed=60.0, label="继续行驶")
    data.append(("VIN_1_DRIVE", t, 0, 0.0, "下电熄火"))

    # 🚗 [测试 2] 前驻车切分测试 (原地热车 > 15分)
    t = datetime.datetime(2023, 10, 2, 9, 0, 0)
    t = append_action(data, "VIN_2_PRE_SPLIT", t, duration_minutes=20, power=2, speed=0.0, label="上电静止(20分)")
    t = append_action(data, "VIN_2_PRE_SPLIT", t, duration_minutes=5, power=2, speed=30.0, label="起步")
    data.append(("VIN_2_PRE_SPLIT", t, 0, 0.0, "下电熄火"))

    # 🚗 [测试 3] 后驻车切分测试 (停车不熄火 > 15分)
    t = datetime.datetime(2023, 10, 2, 11, 0, 0)
    t = append_action(data, "VIN_3_POST_SPLIT", t, duration_minutes=10, power=2, speed=40.0, label="行驶")
    t = append_action(data, "VIN_3_POST_SPLIT", t, duration_minutes=20, power=2, speed=0.0, label="停车不熄火(20分)")
    data.append(("VIN_3_POST_SPLIT", t, 0, 0.0, "终于下电"))

    return data

# ==========================================
# 3. 执行计算与对比展示
# ==========================================
raw_records = generate_core_logic_data()
raw_df = spark.createDataFrame(
    raw_records, 
    schema=["vin", "timestamp", "syspowermod_2012001", "vehspd_2011002", "action_label"]
)

# 剥离 action_label 投入 UDF (UDF不需要这个字段，这只是为了打印好看)
input_df = raw_df.select(
    F.col("vin"), F.col("timestamp"), F.col("syspowermod_2012001"), F.col("vehspd_2011002"),
    F.lit(0).cast(IntegerType()).alias("is_state"), F.lit(None).cast(StringType()).alias("state_phase"),
    F.lit(None).cast(TimestampType()).alias("state_trip_start"), F.lit(None).cast(TimestampType()).alias("state_stop_start"), F.lit(None).cast(StringType()).alias("state_speeds")
)

processed_df = input_df.groupBy("vin").applyInPandas(process_t_plus_1_with_metrics, schema=result_schema).cache()
trips_df = processed_df.filter(F.col("row_type") == 'TRIP')

# 为了不被 1秒1条 的数据刷屏，我们用 groupBy 浓缩打印原始输入：
raw_summary_df = raw_df.groupBy("vin", "action_label").agg(
    F.min("timestamp").alias("动作开始时间"),
    F.max("timestamp").alias("动作结束时间"),
    F.round(F.avg("vehspd_2011002"), 1).alias("大概车速")
).orderBy("vin", "动作开始时间")

# ----------------- 打印区域 -----------------
vins = ["VIN_1_DRIVE", "VIN_2_PRE_SPLIT", "VIN_3_POST_SPLIT"]

print("\n" + "★"*80)
print(" 🚀 V2X 行程切分逻辑：输入与输出对比验证")
print("★"*80)

for v in vins:
    print(f"\n\n=======================================================")
    print(f" 🚘 当前车辆: {v}")
    print(f"=======================================================")
    
    print("\n[A. 切分前的原始动作序列 (已浓缩)]")
    raw_summary_df.filter(F.col("vin") == v).select(
        "action_label", 
        F.date_format("动作开始时间", "HH:mm:ss").alias("开始"), 
        F.date_format("动作结束时间", "HH:mm:ss").alias("结束"), 
        "大概车速"
    ).show(truncate=False)

    print("\n[B. 切分后的行程结果表 (TRIP)]")
    trips_df.filter(F.col("vin") == v).select(
        "trip_type", 
        F.date_format("start_time", "HH:mm:ss").alias("开始时间"), 
        F.date_format("end_time", "HH:mm:ss").alias("结束时间"),
        F.round("max_speed", 2).alias("最高时速"),
        F.round("avg_speed", 2).alias("平均时速")
    ).orderBy("开始时间").show(truncate=False)

spark.stop()