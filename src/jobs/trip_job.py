import argparse
import pandas as pd
import numpy as np
from typing import Tuple, Iterator
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

# ==========================================
# 1. 定义 Schema
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
# 2. 核心状态机算法 (EventTime 驱动)
# ==========================================
def process_unified_trip_state(
    key: Tuple[str], 
    pdf_iter: Iterator[pd.DataFrame], 
    state: GroupState
) -> Iterator[pd.DataFrame]:
    
    vin = key[0]
    output_rows = []

    # --- 1. 从 State 恢复车辆历史记忆 ---
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

    # --- 2. 处理水位线超时 (断网/长期离线兜底结算) ---
    if state.hasTimedOut:
        if phase != 'OFF' and trip_start is not None:
            end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
            trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
            if trip_record: 
                output_rows.append(trip_record)
        
        state.remove()
        if output_rows:
            yield pd.DataFrame(output_rows, columns=RESULT_SCHEMA.names)
        return

    # --- 3. 消费微批次新流入数据 ---
    for pdf in pdf_iter:
        pdf = pdf.sort_values(by=['timestamp']) # 离线跑批必须排序，防止洗牌乱序
        
        for _, row in pdf.iterrows():
            ts = row['timestamp']
            power = row['syspowermod_2012001']
            speed = float(row['vehspd_2011002']) if pd.notna(row['vehspd_2011002']) else 0.0
            
            # [突发断层检查]：应对单批次内时间直接跳跃超过15分钟的脏数据
            if phase != 'OFF' and last_ts is not None:
                if (ts - last_ts).total_seconds() > 900: 
                    end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                    trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    phase, trip_speeds = 'OFF', []
                    trip_type_so_far, last_speed_ts = 'PARKING', None

            # ================= 状态机核心跃迁 =================
            if phase == 'OFF':
                # 触发上电 或 速度>0 -> 开启新行程
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_start = ts
                    trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                    last_speed_ts = ts if speed > 0 else None
                    trip_speeds = [speed]

            elif phase == 'HANG_OFF':
                # 👉 【修改点 1】：这里的 300 秒（5分钟）改成了 900 秒（15分钟）
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
                # 👉 【修改点 2】：删除了这里的 "如果 speed==0 且长达15分钟就截断" 的逻辑。
                # 现在的逻辑非常干净：只要是 ACTIVE，就单纯收集速度
                trip_speeds.append(speed)
                if speed > 0: 
                    trip_type_so_far = 'DRIVING'
                    last_speed_ts = ts

                # 接收到下电信号 -> 进入【挂起/观察期】
                if power == 0:
                    phase = 'HANG_OFF'
                    hang_start = ts

            last_ts = ts

    # --- 4. 提交状态并设置下一次超时唤醒“闹钟” ---
    if phase != 'OFF' and last_ts is not None:
        state_speeds_str = ','.join(map(str, trip_speeds)) if trip_speeds else ""
        state.update((
            phase, trip_type_so_far, trip_start, hang_start, 
            last_ts, last_speed_ts, state_speeds_str
        ))
        
        # 👉 【修改点 3】：将下电挂起的兜底闹钟，从 5 分钟延长到了 15 分钟
        if phase == 'HANG_OFF':
            timeout_timestamp_ms = int(hang_start.timestamp() * 1000) + (15 * 60 * 1000)
        else:
            timeout_timestamp_ms = int(last_ts.timestamp() * 1000) + (15 * 60 * 1000)
            
        state.setTimeoutTimestamp(timeout_timestamp_ms)
    else:
        state.remove()

    if output_rows:
        yield pd.DataFrame(output_rows, columns=RESULT_SCHEMA.names)
    else:
        yield pd.DataFrame(columns=RESULT_SCHEMA.names)

# ==========================================
# 3. 双模流转 Pipeline (Kafka 实时 & Paimon 离线)
# ==========================================
def run_trip_engine(spark: SparkSession, run_mode: str):
    
    checkpoint_dir = f"/path/to/telematics_data_warehouse/checkpoints/trip_{run_mode}"
    
    if run_mode == "offline":
        raw_stream = spark.readStream \
            .format("paimon") \
            .load("paimon_catalog.ods_telematics.ods_veh_log")
            
        trigger_conf = {"availableNow": True} 
        
    elif run_mode == "realtime":
        raw_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "10.0.0.1:9092,10.0.0.2:9092") \
            .option("subscribe", "dwd_vehicle_logs_topic") \
            .option("startingOffsets", "latest") \
            .load() \
            .selectExpr("CAST(value AS STRING) as json_str")
        
        # TODO: 依据实际情况将 JSON 解析为 DataFrame 列
        
        trigger_conf = {"processingTime": "10 seconds"}

    enriched_stream = raw_stream \
        .withWatermark("timestamp", "0 minutes") \
        .groupBy("vin") \
        .applyInPandasWithState(
            func=process_unified_trip_state,
            outputStructType=RESULT_SCHEMA,
            stateStructType=STATE_SCHEMA,
            outputMode="append",
            timeoutConf=GroupStateTimeout.EventTimeTimeout
        )

    def write_to_bytehouse(batch_df, batch_id):
        valid_trips_df = batch_df.filter(F.col("row_type") == 'TRIP')
        
        if not valid_trips_df.isEmpty():
            valid_trips_df.write \
                .format("jdbc") \
                .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
                .option("url", "jdbc:bytehouse://your-bytehouse-endpoint:8123") \
                .option("dbtable", "ads_telematics.ads_trip_summary") \
                .mode("append") \
                .save()

    query = enriched_stream.writeStream \
        .foreachBatch(write_to_bytehouse) \
        .option("checkpointLocation", checkpoint_dir) \
        .trigger(**trigger_conf) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="telematics_data_warehouse_trip_engine")
    parser.add_argument("--mode", required=True, choices=['realtime', 'offline'], help="运行模式: realtime 或 offline")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName(f"telematics_data_warehouse_trip_engine_{args.mode}") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    run_trip_engine(spark, args.mode)