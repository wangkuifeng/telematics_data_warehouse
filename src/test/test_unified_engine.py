import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Iterator
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

# ==========================================
# 1. 核心 Schema 定义
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

STATE_SCHEMA = StructType([
    StructField("phase", StringType()),
    StructField("trip_type_so_far", StringType()),
    StructField("trip_start", TimestampType()),
    StructField("hang_start", TimestampType()),
    StructField("last_ts", TimestampType()),
    StructField("last_speed_ts", TimestampType()),
    StructField("state_speeds", StringType())
])

# ==========================================
# 2. 状态机引擎核心代码 (彻底修复时区与Yield问题)
# ==========================================
def process_unified_trip_state(
    key: Tuple[str], 
    pdf_iter: Iterator[pd.DataFrame], 
    state: GroupState
) -> Iterator[pd.DataFrame]:
    
    vin = key[0]
    output_rows = []

    if state.exists:
        state_data = state.get
        phase = state_data[0]
        trip_type_so_far = state_data[1]
        trip_start = state_data[2]
        hang_start = state_data[3]
        last_ts = state_data[4]
        last_speed_ts = state_data[5]
        speeds_str = state_data[6]
        trip_speeds = [float(x) for x in speeds_str.split(',')] if speeds_str else []
    else:
        phase, trip_type_so_far = 'OFF', 'PARKING'
        trip_start, hang_start, last_ts, last_speed_ts = None, None, None, None
        trip_speeds = []

    def emit_trip(t_type, t_start, t_end, speeds):
        if pd.isna(t_start) or pd.isna(t_end) or t_start >= t_end:
            return None
        final_type = 'DRIVING' if any(s > 0 for s in speeds) else 'PARKING'
        max_s = float(np.max(speeds)) if speeds else 0.0
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        return (vin, 'TRIP', final_type, None, t_start, t_end, max_s, avg_s, t_end, None, None)

    # 👉 修复1: 发生超时时，必须保证有且仅有一个 DataFrame 被 Yield 出来
    if state.hasTimedOut:
        if phase != 'OFF' and trip_start is not None:
            end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
            trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
            if trip_record: 
                output_rows.append(trip_record)
        state.remove()
        
        if output_rows:
            yield pd.DataFrame(output_rows, columns=RESULT_SCHEMA.names)
        else:
            yield pd.DataFrame(columns=RESULT_SCHEMA.names) # 核心修复：防止静默失败
        return

    for pdf in pdf_iter:
        pdf = pdf.sort_values(by=['timestamp']) 
        
        for _, row in pdf.iterrows():
            ts = row['timestamp']
            power = row['syspowermod_2012001']
            speed = float(row['vehspd_2011002']) if pd.notna(row['vehspd_2011002']) else 0.0
            
            if phase != 'OFF' and last_ts is not None:
                if (ts - last_ts).total_seconds() > 900: 
                    end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                    trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    phase, trip_speeds = 'OFF', []
                    trip_type_so_far, last_speed_ts = 'PARKING', None

            if phase == 'OFF':
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_start = ts
                    trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                    last_speed_ts = ts if speed > 0 else None
                    trip_speeds = [speed]

            elif phase == 'HANG_OFF':
                if (ts - hang_start).total_seconds() > 900: 
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
                trip_speeds.append(speed)
                if speed > 0: 
                    trip_type_so_far = 'DRIVING'
                    last_speed_ts = ts

                if power == 0:
                    phase = 'HANG_OFF'
                    hang_start = ts

            last_ts = ts

    if phase != 'OFF' and last_ts is not None:
        state_speeds_str = ','.join(map(str, trip_speeds)) if trip_speeds else ""
        state.update((
            phase, trip_type_so_far, trip_start, hang_start, 
            last_ts, last_speed_ts, state_speeds_str
        ))
        
        # 👉 修复2: 废弃 timestamp()，改用 pandas 底层纳秒值强制对齐，彻底消除时区漂移
        if phase == 'HANG_OFF':
            timeout_timestamp_ms = int(hang_start.value // 1000000) + (15 * 60 * 1000)
        else:
            timeout_timestamp_ms = int(last_ts.value // 1000000) + (15 * 60 * 1000)
            
        state.setTimeoutTimestamp(timeout_timestamp_ms)
    else:
        state.remove()

    if output_rows:
        yield pd.DataFrame(output_rows, columns=RESULT_SCHEMA.names)
    else:
        yield pd.DataFrame(columns=RESULT_SCHEMA.names)

# ==========================================
# 3. 构造测试数据
# ==========================================
def generate_mock_data_in_chunks(output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    base_time = datetime(2026, 10, 1, 8, 0, 0)
    
    # --- Chunk 1: 真实车辆轨迹 (时间范围 8:00 ~ 8:21) ---
    data_chunk_1 = []
    
    # 场景 A: 正常通勤 
    vin = "VIN_001_NORMAL"
    data_chunk_1.append({"vin": vin, "timestamp": base_time, "syspowermod_2012001": 2, "vehspd_2011002": 0.0})
    for i in range(1, 10):
        data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=i), "syspowermod_2012001": 2, "vehspd_2011002": 40.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=10), "syspowermod_2012001": 2, "vehspd_2011002": 0.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=11), "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # 场景 B: 长时间怠速不熄火
    vin = "VIN_002_LONG_IDLE"
    data_chunk_1.append({"vin": vin, "timestamp": base_time, "syspowermod_2012001": 2, "vehspd_2011002": 30.0})
    for i in range(1, 20): 
        data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=i), "syspowermod_2012001": 2, "vehspd_2011002": 0.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=20), "syspowermod_2012001": 2, "vehspd_2011002": 20.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=21), "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # 场景 C: 加油站短时熄火防抖 
    vin = "VIN_003_SHORT_STOP"
    data_chunk_1.append({"vin": vin, "timestamp": base_time, "syspowermod_2012001": 2, "vehspd_2011002": 50.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=5), "syspowermod_2012001": 0, "vehspd_2011002": 0.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=13), "syspowermod_2012001": 2, "vehspd_2011002": 30.0}) 
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=20), "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # 场景 D: 地下车库断网
    vin = "VIN_004_DISCONNECT"
    data_chunk_1.append({"vin": vin, "timestamp": base_time, "syspowermod_2012001": 2, "vehspd_2011002": 30.0})
    data_chunk_1.append({"vin": vin, "timestamp": base_time + timedelta(minutes=5), "syspowermod_2012001": 2, "vehspd_2011002": 15.0})

    # --- Chunk 2-4: 未来时间刺客，暴力推进 Watermark ---
    data_chunk_2 = [{"vin": "VIN_999_PUSHER", "timestamp": base_time + timedelta(hours=1), "syspowermod_2012001": 2, "vehspd_2011002": 60.0}]
    data_chunk_3 = [{"vin": "VIN_999_PUSHER", "timestamp": base_time + timedelta(hours=2), "syspowermod_2012001": 0, "vehspd_2011002": 0.0}]
    data_chunk_4 = [{"vin": "VIN_999_PUSHER", "timestamp": base_time + timedelta(hours=3), "syspowermod_2012001": 0, "vehspd_2011002": 0.0}]

    spark = SparkSession.builder.getOrCreate()
    input_schema = StructType([
        StructField("vin", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("syspowermod_2012001", IntegerType()),
        StructField("vehspd_2011002", DoubleType())
    ])

    # 写 4 个文件，触发 4 个微批次
    spark.createDataFrame(data_chunk_1, schema=input_schema).coalesce(1).write.mode("append").parquet(output_dir)
    spark.createDataFrame(data_chunk_2, schema=input_schema).coalesce(1).write.mode("append").parquet(output_dir)
    spark.createDataFrame(data_chunk_3, schema=input_schema).coalesce(1).write.mode("append").parquet(output_dir)
    spark.createDataFrame(data_chunk_4, schema=input_schema).coalesce(1).write.mode("append").parquet(output_dir)
    
    return input_schema

# ==========================================
# 4. 执行验证的主程序
# ==========================================
def run_local_test():
    spark = SparkSession.builder \
        .appName("telematics_local_test") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    mock_data_dir = "/tmp/telematics_mock_ods"
    checkpoint_dir = "/tmp/telematics_checkpoints"

    # 强制清理环境
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        
    input_schema = generate_mock_data_in_chunks(mock_data_dir)

    print("\n🚀 开始运行离线引擎验证...")
    
    raw_stream = spark.readStream \
        .schema(input_schema) \
        .option("maxFilesPerTrigger", 1) \
        .parquet(mock_data_dir)

    enriched_stream = raw_stream \
        .withWatermark("timestamp", "5 minutes") \
        .groupBy("vin") \
        .applyInPandasWithState(
            func=process_unified_trip_state,
            outputStructType=RESULT_SCHEMA,
            stateStructType=STATE_SCHEMA,
            outputMode="append",
            timeoutConf=GroupStateTimeout.EventTimeTimeout
        )

    # 打印到控制台
    query = enriched_stream \
        .filter("vin != 'VIN_999_PUSHER'") \
        .writeStream \
        .format("console") \
        .option("truncate", "false") \
        .option("checkpointLocation", checkpoint_dir) \
        .trigger(availableNow=True) \
        .start()

    query.awaitTermination()
    print("\n✅ 所有数据处理完毕！")

if __name__ == "__main__":
    run_local_test()