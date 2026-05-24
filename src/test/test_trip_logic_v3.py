import pandas as pd
import numpy as np
import datetime
import hashlib
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

# ==========================================
# 0. 初始化 Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("V2X_Trip_Raw_Data_Visible") \
    .master("local[*]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

result_schema = StructType([
    StructField("trip_id", StringType()),       
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
# 1. 核心 UDF (逻辑保持不变)
# ==========================================
def process_trip_with_id(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
    vin = key[0]
    output_rows = []
    
    pdf = pdf.sort_values(by=['timestamp', 'is_state'], ascending=[True, False])
    
    phase, trip_start, stop_start = 'OFF', None, None
    last_ts = None
    trip_speeds = [] 
    SPEED_THRESHOLD = 3.0
    MAX_GAP_SECONDS = 12 * 3600 
    
    def emit_trip(trip_t, t_start, t_end, speeds):
        if not speeds: speeds = [0.0]
        s_max = float(np.max(speeds))
        s_avg = float(np.mean(speeds)) 
        raw_str = f"{vin}_{t_start.strftime('%Y%m%d%H%M%S')}"
        trip_id = hashlib.md5(raw_str.encode('utf-8')).hexdigest()
        return (trip_id, vin, 'TRIP', trip_t, t_start, t_end, s_max, s_avg, t_end, None)

    for _, row in pdf.iterrows():
        if row['is_state'] == 1:
            phase, trip_start, stop_start = row['state_phase'], row['state_trip_start'], row['state_stop_start']
            last_ts, state_speeds_str = row['timestamp'], row.get('state_speeds', '')
            trip_speeds = [float(x) for x in state_speeds_str.split(',')] if pd.notna(state_speeds_str) and state_speeds_str else []
            continue
            
        ts, power, speed = row['timestamp'], row['syspowermod_2012001'], row['vehspd_2011002']
        
        if last_ts and (ts - last_ts).total_seconds() > MAX_GAP_SECONDS:
            if phase in ['DRIVE', 'STOP_WAIT']:
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
                if (ts - trip_start).total_seconds() / 60.0 > 15:
                    output_rows.append(emit_trip('PRE_PARKING', trip_start, ts, []))
                    trip_start, trip_speeds = ts, [speed]
                else:
                    trip_speeds.append(speed)
                phase = 'DRIVE'
            elif power == 0: phase = 'OFF'

        elif phase == 'DRIVE':
            if speed > SPEED_THRESHOLD: trip_speeds.append(speed)
            if speed <= SPEED_THRESHOLD or power == 0:
                phase, stop_start = 'STOP_WAIT', ts

        elif phase == 'STOP_WAIT':
            if speed > SPEED_THRESHOLD: trip_speeds.append(speed)
            duration_mins = (ts - stop_start).total_seconds() / 60.0
            
            if speed > SPEED_THRESHOLD and power == 2:
                if duration_mins <= 15: phase = 'DRIVE'
                else:
                    output_rows.append(emit_trip('DRIVING', trip_start, stop_start, trip_speeds[:-1]))
                    output_rows.append(emit_trip('POST_PARKING', stop_start, ts, [0.0]))
                    phase, trip_start, trip_speeds = 'DRIVE', ts, [speed]
            elif power == 0:
                if duration_mins <= 15:
                    output_rows.append(emit_trip('DRIVING', trip_start, ts, trip_speeds))
                else:
                    output_rows.append(emit_trip('DRIVING', trip_start, stop_start, trip_speeds[:-1]))
                    output_rows.append(emit_trip('POST_PARKING', stop_start, ts, [0.0]))
                phase = 'OFF'

    # EOF 离线尾部强制结算
    if phase == 'DRIVE':
        output_rows.append(emit_trip('DRIVING', trip_start, last_ts, trip_speeds))
    elif phase == 'STOP_WAIT':
        drive_speeds = trip_speeds[:-1] if len(trip_speeds) > 1 else trip_speeds
        output_rows.append(emit_trip('DRIVING', trip_start, stop_start, drive_speeds))
        if stop_start < last_ts:
            output_rows.append(emit_trip('POST_PARKING', stop_start, last_ts, [0.0]))
    elif phase == 'PRE_PARK':
        output_rows.append(emit_trip('PRE_PARKING', trip_start, last_ts, trip_speeds))
        
    state_speeds_str = ','.join(map(str, trip_speeds)) if phase != 'OFF' and trip_speeds else None
    output_rows.append((None, vin, 'STATE', phase, trip_start, stop_start, None, None, last_ts, state_speeds_str))

    return pd.DataFrame(output_rows, columns=result_schema.names)

# ==========================================
# 2. 模拟底盘数据 (修改为 1分钟1条，所见即所得)
# ==========================================
def append_action(data_list, vin, start_time, duration_minutes, power, speed_base, remark):
    """按 1分钟 间隔生成明细数据，加上备注方便阅读"""
    current_time = start_time
    for _ in range(duration_minutes):
        # 加上微小随机波动，避免速度全是死数
        actual_speed = round(speed_base + np.random.uniform(-1.5, 1.5), 1) if speed_base > 0 else 0.0
        data_list.append((vin, current_time, power, actual_speed, remark))
        current_time += datetime.timedelta(minutes=1)
    return current_time

data = []
t = datetime.datetime(2023, 10, 2, 8, 0, 0)

# [场景流转]
# 1. 正常行驶 5 分钟
t = append_action(data, "VIN_RAW_TEST", t, duration_minutes=5, power=2, speed_base=40.0, remark="行驶中") 
# 2. 遇到红绿灯，踩刹车停住 2 分钟 (上电=2, 车速=0)
t = append_action(data, "VIN_RAW_TEST", t, duration_minutes=2, power=2, speed_base=0.0,  remark="等红绿灯(未熄火)")   
# 3. 绿灯亮起，继续行驶 5 分钟
t = append_action(data, "VIN_RAW_TEST", t, duration_minutes=5, power=2, speed_base=60.0, remark="继续行驶") 
# 4. 到达目的地，彻底下电熄火 20 分钟 (触发 15 分钟切分规则)
t = append_action(data, "VIN_RAW_TEST", t, duration_minutes=20, power=0, speed_base=0.0, remark="停车并下电熄火") 

raw_df = spark.createDataFrame(data, schema=["vin", "timestamp", "syspowermod", "vehspd", "remark"])

# 准备 UDF 输入
input_df = raw_df.select(
    F.col("vin"), F.col("timestamp"), F.col("syspowermod").alias("syspowermod_2012001"), F.col("vehspd").alias("vehspd_2011002"),
    F.lit(0).cast(IntegerType()).alias("is_state"), F.lit(None).cast(StringType()).alias("state_phase"),
    F.lit(None).cast(TimestampType()).alias("state_trip_start"), F.lit(None).cast(TimestampType()).alias("state_stop_start"), 
    F.lit(None).cast(StringType()).alias("state_speeds")
)

trips_df = input_df.groupBy("vin").applyInPandas(process_trip_with_id, schema=result_schema) \
    .filter(F.col("row_type") == 'TRIP').cache()

# ==========================================
# 3. 模拟事件数据并关联
# ==========================================
events_data = [
    ("VIN_RAW_TEST", "WINDOW_OPEN", datetime.datetime(2023, 10, 2, 8, 2, 0), datetime.datetime(2023, 10, 2, 8, 4, 0)),
    ("VIN_RAW_TEST", "MUSIC_PLAY",  datetime.datetime(2023, 10, 2, 8, 8, 0), datetime.datetime(2023, 10, 2, 8, 11, 0))
]
events_df = spark.createDataFrame(events_data, schema=["vin", "event_type", "event_start", "event_end"])

joined_events_df = events_df.join(
    trips_df,
    (events_df.vin == trips_df.vin) & 
    (events_df.event_start >= trips_df.start_time) & 
    (events_df.event_start <= trips_df.end_time),
    "left"
).select(
    events_df.vin, events_df.event_type, 
    events_df.event_start, events_df.event_end, 
    trips_df.trip_id
)

# ----------------- 打印区域 -----------------
print("\n" + "="*80)
print(" [A. 喂给 UDF 的最原始明细数据 (1分钟1条，无任何折叠)]")
print("="*80)
# 直接 show 展现每一行真实的样子
raw_df.select(
    "vin", 
    F.date_format("timestamp", "HH:mm:ss").alias("时间戳"), 
    "syspowermod", 
    "vehspd", 
    "remark"
).show(50, truncate=False)

print("\n" + "="*80)
print(" [B. 经过 UDF 吐出的宏观行程主表]")
print("="*80)
trips_df.select(
    "trip_id", "trip_type", 
    F.date_format("start_time", "HH:mm:ss").alias("trip_start"), 
    F.date_format("end_time", "HH:mm:ss").alias("trip_end"),
    F.round("max_speed", 1).alias("max_spd"),
    F.round("avg_speed", 1).alias("avg_spd")
).show(truncate=False)

print("\n" + "="*80)
print(" [C. 事件子表 (成功关联了 Trip ID)]")
print("="*80)
joined_events_df.select(
    "event_type", 
    F.date_format("event_start", "HH:mm:ss").alias("开始"), 
    F.date_format("event_end", "HH:mm:ss").alias("结束"),
    "trip_id"
).show(truncate=False)

spark.stop()