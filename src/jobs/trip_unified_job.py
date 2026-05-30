import argparse
import pandas as pd
import numpy as np
import json
import hashlib  # [新增] 用于生成 trip_id
from typing import Tuple, Iterator
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

# ==========================================
# 1. 定义结果与状态 Schema
# ==========================================
# 最终结果表 Schema：[新增] 首位增加 trip_id 字段
RESULT_SCHEMA = StructType([
    StructField("trip_id", StringType()),       # [新增] 行程唯一标识 (MD5)
    StructField("vin", StringType()),
    StructField("row_type", StringType()),      # 'TRIP' 或 'EVENT' (后续入库前可丢弃)
    StructField("trip_type", StringType()),     # 行程类型 (DRIVING/PARKING) 或 二级事件名 (如 CREEP_RUN_EVENT)
    StructField("phase", StringType()),         
    StructField("start_time", TimestampType()), 
    StructField("end_time", TimestampType()),   
    StructField("max_speed", DoubleType()),     
    StructField("avg_speed", DoubleType()),     
    StructField("start_lon", DoubleType()),     
    StructField("start_lat", DoubleType()),     
    StructField("end_lon", DoubleType()),       
    StructField("end_lat", DoubleType()),       
    StructField("last_ts", TimestampType()),    
    StructField("last_speed_ts", TimestampType()), 
    StructField("state_speeds", StringType())   
])

# 状态机内部记忆 Schema (保持完全兼容原版结构，不增加额外内存开销)
STATE_SCHEMA = StructType([
    StructField("phase", StringType()),
    StructField("trip_type_so_far", StringType()),
    StructField("trip_start", TimestampType()),
    StructField("hang_start", TimestampType()),
    StructField("last_ts", TimestampType()),
    StructField("last_speed_ts", TimestampType()),
    StructField("state_speeds", StringType()),
    StructField("ongoing_events", StringType()),      
    StructField("first_speed_emitted", BooleanType()) 
])

# ==========================================
# 2. 核心状态机算法 (支持多事件、多维坐标与主从ID关联)
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
        
        try:
            ongoing_events = json.loads(state_data[7]) if state_data[7] else {}
        except Exception:
            ongoing_events = {}
        first_speed_emitted = state_data[8] if state_data[8] is not None else False
    else:
        phase, trip_type_so_far = 'OFF', 'PARKING'
        trip_start, hang_start, last_ts, last_speed_ts = None, None, None, None
        trip_speeds = []
        ongoing_events = {}
        first_speed_emitted = False

    # --- [新增] MD5 生成辅助函数 ---
    def get_trip_id(v_id, t_start):
        if pd.isna(t_start): return None
        raw_str = f"{v_id}_{t_start}"
        return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

    # --- 基础发射器：主行程 [修改：增加 trip_id] ---
    def emit_trip(t_type, t_start, t_end, speeds):
        if pd.isna(t_start) or pd.isna(t_end) or t_start >= t_end:
            return None
        final_type = 'DRIVING' if any(s > 0 for s in speeds) else 'PARKING'
        max_s = float(np.max(speeds)) if speeds else 0.0
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        
        current_trip_id = get_trip_id(vin, t_start)
        
        return (current_trip_id, vin, 'TRIP', final_type, None, t_start, t_end, max_s, avg_s, None, None, None, None, t_end, None, None)

    # --- 基础发射器：二级事件 [修改：增加 trip_id 映射逻辑与 parent_trip_start 参数] ---
    def emit_sub_event(event_name, evt_start, evt_end, s_lon, s_lat, e_lon, e_lat, parent_trip_start):
        if pd.isna(evt_start) or pd.isna(evt_end) or evt_start >= evt_end:
            return None
        
        # 容错：如果发生游离事件，使用事件自己的时间作为备用ID生成逻辑
        effective_trip_start = parent_trip_start if parent_trip_start is not None else evt_start
        current_trip_id = get_trip_id(vin, effective_trip_start)
        
        return (current_trip_id, vin, 'EVENT', event_name, None, evt_start, evt_end, 0.0, 0.0, s_lon, s_lat, e_lon, e_lat, evt_end, None, None)

    # --- [修改：增加 current_trip_start 传参，以便将所有事件挂载到父行程] ---
    def flush_all_ongoing_events(end_time, current_lon, current_lat, current_trip_start):
        nonlocal ongoing_events
        # 1. 结算：驻留事件 (PARK_AFTER_RUN_EVENT)
        if last_speed_ts is not None and last_speed_ts < end_time:
            evt = emit_sub_event('PARK_AFTER_RUN_EVENT', last_speed_ts, end_time, current_lon, current_lat, current_lon, current_lat, current_trip_start)
            if evt: output_rows.append(evt)
            
        # 2. 结算：其余在字典中未闭合的持续事件
        for evt_name, evt_ctx in list(ongoing_events.items()):
            start_ts_pd = pd.to_datetime(evt_ctx["start_time"])
            s_lon = evt_ctx["start_lon"]
            s_lat = evt_ctx["start_lat"]
            duration = (end_time - start_ts_pd).total_seconds()
            
            min_dur = 300 if evt_name in ['CREEP_RUN_EVENT', 'TRAFFIC_JAM_EVENT'] else (60 if 'SEAT' in evt_name or 'TRUNK' in evt_name else 0)
            if duration >= min_dur:
                evt = emit_sub_event(evt_name, start_ts_pd, end_time, s_lon, s_lat, current_lon, current_lat, current_trip_start)
                if evt: output_rows.append(evt)
        ongoing_events.clear()

    # --- 2. 处理水位线事件时间超时 (断网/跨天结算兜底) ---
    if state.hasTimedOut:
        if phase != 'OFF' and trip_start is not None:
            end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
            trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
            if trip_record: output_rows.append(trip_record)
            
            # [修改] 透传 trip_start
            flush_all_ongoing_events(end_time_fallback, 0.0, 0.0, trip_start)
        
        state.remove()
        if output_rows:
            yield pd.DataFrame(output_rows, columns=RESULT_SCHEMA.names)
        return

    # --- 3. 逐行消费微批次/全天排序数据 ---
    for pdf in pdf_iter:
        pdf = pdf.sort_values(by=['timestamp']) 
        
        for _, row in pdf.iterrows():
            ts = row['timestamp']
            power = row.get('syspowermod_2012001', 0)
            speed = float(row.get('vehspd_2011002', 0.0)) if pd.notna(row.get('vehspd_2011002')) else 0.0
            
            lon = float(row.get('gps_longitude', 0.0)) if pd.notna(row.get('gps_longitude')) else 0.0
            lat = float(row.get('gps_latitude', 0.0)) if pd.notna(row.get('gps_latitude')) else 0.0
            
            # [时间异常裂断防御] 
            if phase != 'OFF' and last_ts is not None:
                if (ts - last_ts).total_seconds() > 900: 
                    end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                    trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    
                    # [修改] 透传 trip_start
                    flush_all_ongoing_events(end_time_fallback, lon, lat, trip_start)
                    
                    phase, trip_speeds = 'OFF', []
                    trip_type_so_far, last_speed_ts = 'PARKING', None
                    first_speed_emitted = False

            # ================= 一级主状态机流转 =================
            if phase == 'OFF':
                if power == 2 or speed > 0:
                    phase = 'ACTIVE'
                    trip_start = ts
                    trip_type_so_far = 'DRIVING' if speed > 0 else 'PARKING'
                    last_speed_ts = ts if speed > 0 else None
                    trip_speeds = [speed]
                    first_speed_emitted = False 

            elif phase == 'HANG_OFF':
                if (ts - hang_start).total_seconds() > 900: 
                    trip_record = emit_trip(trip_type_so_far, trip_start, hang_start, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    
                    # [修改] 透传 trip_start
                    flush_all_ongoing_events(hang_start, lon, lat, trip_start)
                    
                    phase, trip_speeds = 'OFF', []
                    trip_type_so_far, last_speed_ts = 'PARKING', None
                    first_speed_emitted = False
                    
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
                    
                    # 结算：驾乘准备事件 (PARK_BEFORE_RUN_EVENT)
                    if not first_speed_emitted and trip_start is not None:
                        # [修改] 增加 parent_trip_start (此处即为 trip_start)
                        evt = emit_sub_event('PARK_BEFORE_RUN_EVENT', trip_start, ts, lon, lat, lon, lat, trip_start)
                        if evt: output_rows.append(evt)
                        first_speed_emitted = True

            # ================= 二级事件检测引擎 (全面覆盖 23 个事件) =================
            if phase == 'ACTIVE':
                current_conditions = {
                    'CREEP_RUN_EVENT': (0 < speed <= 10, 300),
                    'TRAFFIC_JAM_EVENT': (10 < speed <= 20, 300),
                    'LOW_SPEED_EVENT': (0 < speed <= 40, 0),
                    'MEDIUM_SPEED_EVENT': (40 < speed <= 80, 0),
                    'HIGH_SPEED_EVENT': (80 < speed <= 120, 0),
                    'OVER_SPEED_EVENT': (speed > 120, 0),
                    'PARK_RUNNING_EVENT': (speed == 0, 0),
                    
                    'ON_OFF_IVI_EVENT': (row.get('ivi_power_status', 0) == 1, 0),
                    'TRUNK_EVENT': (row.get('trunk_door_status', 0) == 1, 60),
                    
                    'DRIV_SEAT_VENTNSTS_EVENT': (row.get('driv_seat_vent_sts', 0) == 1, 60),
                    'PASS_SEAT_VENTNSTS_EVENT': (row.get('pass_seat_vent_sts', 0) == 1, 60),
                    'SECRL_SEAT_VENTNSTS_EVENT': (row.get('secrl_seat_vent_sts', 0) == 1, 60),
                    'SECRR_SEAT_VENTNSTS_EVENT': (row.get('secrr_seat_vent_sts', 0) == 1, 60),
                    
                    'DRIV_SEAT_HEATSTS_EVENT': (row.get('driv_seat_heat_sts', 0) == 1, 60),
                    'PASS_SEAT_HEATSTS_EVENT': (row.get('pass_seat_heat_sts', 0) == 1, 60),
                    'SECRL_SEAT_HEATSTS_EVENT': (row.get('secrl_seat_heat_sts', 0) == 1, 60),
                    'SECRR_SEAT_HEATSTS_EVENT': (row.get('secrr_seat_heat_sts', 0) == 1, 60),
                    
                    'DRIV_SEAT_MASSG_EVENT': (row.get('driv_seat_massg_sts', 0) == 1, 60),
                    'PASS_SEAT_MASSG_EVENT': (row.get('pass_seat_massg_sts', 0) == 1, 60),
                    'SECRL_SEAT_MASSG_EVENT': (row.get('secrl_seat_massg_sts', 0) == 1, 60),
                    'SECRR_SEAT_MASSG_EVENT': (row.get('secrr_seat_massg_sts', 0) == 1, 60)
                }

                for evt_name, (is_active, min_dur_sec) in current_conditions.items():
                    if is_active:
                        if evt_name not in ongoing_events:
                            ongoing_events[evt_name] = {
                                "start_time": str(ts),
                                "start_lon": lon,
                                "start_lat": lat
                            }
                    else:
                        if evt_name in ongoing_events:
                            evt_ctx = ongoing_events.pop(evt_name)
                            start_ts_pd = pd.to_datetime(evt_ctx["start_time"])
                            s_lon = evt_ctx["start_lon"]
                            s_lat = evt_ctx["start_lat"]
                            
                            duration = (ts - start_ts_pd).total_seconds()
                            if duration >= min_dur_sec:
                                # [修改] 透传 trip_start 作为 parent_trip_start
                                evt = emit_sub_event(evt_name, start_ts_pd, ts, s_lon, s_lat, lon, lat, trip_start)
                                if evt: output_rows.append(evt)

            if phase == 'ACTIVE' and power == 0:
                phase = 'HANG_OFF'
                hang_start = ts

            last_ts = ts

    # --- 4. 序列化状态并更新超时唤醒时间 ---
    if phase != 'OFF' and last_ts is not None:
        state_speeds_str = ','.join(map(str, trip_speeds)) if trip_speeds else ""
        ongoing_events_str = json.dumps(ongoing_events)
        
        state.update((
            phase, trip_type_so_far, trip_start, hang_start, 
            last_ts, last_speed_ts, state_speeds_str,
            ongoing_events_str, first_speed_emitted
        ))
        
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
# 3. 批流一体双模执行 Pipeline 引擎
# ==========================================
def run_trip_engine(spark: SparkSession, run_mode: str):
    checkpoint_dir = f"/path/to/telematics_data_warehouse/checkpoints/trip_{run_mode}"
    
    if run_mode == "offline":
        print(">>> [批处理] 启动离线 T+1 模式，加载历史 Paimon ODS 数据...")
        raw_stream = spark.readStream \
            .format("paimon") \
            .load("paimon_catalog.ods_telematics.ods_veh_log")
        trigger_conf = {"availableNow": True} 
        
    elif run_mode == "realtime":
        print(">>> [流处理] 启动实时流模式，绑定 Kafka DWD 实时主题...")
        raw_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "10.0.0.1:9092,10.0.0.2:9092") \
            .option("subscribe", "dwd_vehicle_logs_topic") \
            .option("startingOffsets", "latest") \
            .load() \
            .selectExpr("CAST(value AS STRING) as json_str")
        trigger_conf = {"processingTime": "10 seconds"}

    enriched_stream = raw_stream \
        .withWatermark("timestamp", "10 minutes") \
        .groupBy("vin") \
        .applyInPandasWithState(
            func=process_unified_trip_state,
            outputStructType=RESULT_SCHEMA,
            stateStructType=STATE_SCHEMA,
            outputMode="append",
            timeoutConf=GroupStateTimeout.EventTimeTimeout
        )

    # [修改] 批量汇聚：利用内存进行并行双分流下沉
    def write_to_bytehouse(batch_df, batch_id):
        batch_df.persist() 
        
        # 拆分 1：一级主行程表 (过滤 TRIP 并丢弃中间过程字段 row_type)
        trip_df = batch_df.filter(F.col("row_type") == 'TRIP').drop("row_type")
        if not trip_df.isEmpty():
            print(f">>> 批次 {batch_id}：下沉 {trip_df.count()} 条主行程至 ads_trip_summary...")
            trip_df.write \
                .format("jdbc") \
                .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
                .option("url", "jdbc:bytehouse://your-bytehouse-endpoint:8123") \
                .option("dbtable", "ads_telematics.ads_trip_summary") \
                .mode("append") \
                .save()

        # 拆分 2：二级事件明细表 (过滤 EVENT 并丢弃中间过程字段 row_type)
        event_df = batch_df.filter(F.col("row_type") == 'EVENT').drop("row_type")
        if not event_df.isEmpty():
            print(f">>> 批次 {batch_id}：下沉 {event_df.count()} 条事件明细至 ads_trip_event_detail...")
            event_df.write \
                .format("jdbc") \
                .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
                .option("url", "jdbc:bytehouse://your-bytehouse-endpoint:8123") \
                .option("dbtable", "ads_telematics.ads_trip_event_detail") \
                .mode("append") \
                .save()
                
        batch_df.unpersist()

    query = enriched_stream.writeStream \
        .foreachBatch(write_to_bytehouse) \
        .option("checkpointLocation", checkpoint_dir) \
        .trigger(**trigger_conf) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="telematics_data_warehouse_trip_engine")
    parser.add_argument("--mode", required=True, choices=['realtime', 'offline'], help="执行选择: realtime 或 offline")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName(f"telematics_data_warehouse_trip_engine_{args.mode}") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false") \
        .getOrCreate()

    run_trip_engine(spark, args.mode)