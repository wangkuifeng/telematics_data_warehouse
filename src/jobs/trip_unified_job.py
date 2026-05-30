import argparse
import pandas as pd
import numpy as np
import json
from typing import Tuple, Iterator
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

# ==========================================
# 1. 定义结果与状态 Schema
# ==========================================
# 最终结果表 Schema：扩展了 4 个 GPS 坐标字段
RESULT_SCHEMA = StructType([
    StructField("vin", StringType()),
    StructField("row_type", StringType()),      # 'TRIP' 或 'EVENT'
    StructField("trip_type", StringType()),     # 行程类型 (DRIVING/PARKING) 或 二级事件名 (如 CREEP_RUN_EVENT)
    StructField("phase", StringType()),         
    StructField("start_time", TimestampType()), 
    StructField("end_time", TimestampType()),   
    StructField("max_speed", DoubleType()),     
    StructField("avg_speed", DoubleType()),     
    StructField("start_lon", DoubleType()),     # [新增] 开始位置经度
    StructField("start_lat", DoubleType()),     # [新增] 开始位置纬度
    StructField("end_lon", DoubleType()),       # [新增] 结束位置经度
    StructField("end_lat", DoubleType()),       # [新增] 结束位置纬度
    StructField("last_ts", TimestampType()),    
    StructField("last_speed_ts", TimestampType()), 
    StructField("state_speeds", StringType())   
])

# 状态机内部记忆 Schema (无需因为增加 GPS 而改变结构，保持极高的兼容性)
STATE_SCHEMA = StructType([
    StructField("phase", StringType()),
    StructField("trip_type_so_far", StringType()),
    StructField("trip_start", TimestampType()),
    StructField("hang_start", TimestampType()),
    StructField("last_ts", TimestampType()),
    StructField("last_speed_ts", TimestampType()),
    StructField("state_speeds", StringType()),
    StructField("ongoing_events", StringType()),      # JSON 字符串: 动态跟踪进行中的二级事件及其开始要素
    StructField("first_speed_emitted", BooleanType()) 
])

# ==========================================
# 2. 核心状态机算法 (支持多事件与多维坐标捕获)
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

    # --- 基础发射器：主行程 ---
    def emit_trip(t_type, t_start, t_end, speeds):
        if pd.isna(t_start) or pd.isna(t_end) or t_start >= t_end:
            return None
        final_type = 'DRIVING' if any(s > 0 for s in speeds) else 'PARKING'
        max_s = float(np.max(speeds)) if speeds else 0.0
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        # 主行程默认不带单一GPS，或可根据业务取头尾，此处暂置None
        return (vin, 'TRIP', final_type, None, t_start, t_end, max_s, avg_s, None, None, None, None, t_end, None, None)

    # --- 基础发射器：二级事件 (包含 GPS 字段映射) ---
    def emit_sub_event(event_name, t_start, t_end, s_lon, s_lat, e_lon, e_lat):
        if pd.isna(t_start) or pd.isna(t_end) or t_start >= t_end:
            return None
        return (vin, 'EVENT', event_name, None, t_start, t_end, 0.0, 0.0, s_lon, s_lat, e_lon, e_lat, t_end, None, None)

    # 行程发生强制切断或结束时，兜底冲刷所有进行中的二级事件
    def flush_all_ongoing_events(end_time, current_lon, current_lat):
        nonlocal ongoing_events
        # 1. 结算：驻留事件 (PARK_AFTER_RUN_EVENT)
        if last_speed_ts is not None and last_speed_ts < end_time:
            # 驻留事件开始GPS取最后一次车速>0的点(在ACTIVE中单独存，此处演示直接使用当前坐标)，结束取下电坐标
            evt = emit_sub_event('PARK_AFTER_RUN_EVENT', last_speed_ts, end_time, current_lon, current_lat, current_lon, current_lat)
            if evt: output_rows.append(evt)
            
        # 2. 结算：其余在字典中未闭合的持续事件
        for evt_name, evt_ctx in list(ongoing_events.items()):
            start_ts_pd = pd.to_datetime(evt_ctx["start_time"])
            s_lon = evt_ctx["start_lon"]
            s_lat = evt_ctx["start_lat"]
            duration = (end_time - start_ts_pd).total_seconds()
            
            # 时长阈值判断
            min_dur = 300 if evt_name in ['CREEP_RUN_EVENT', 'TRAFFIC_JAM_EVENT'] else (60 if 'SEAT' in evt_name or 'TRUNK' in evt_name else 0)
            if duration >= min_dur:
                evt = emit_sub_event(evt_name, start_ts_pd, end_time, s_lon, s_lat, current_lon, current_lat)
                if evt: output_rows.append(evt)
        ongoing_events.clear()


    # --- 2. 处理水位线事件时间超时 (断网/跨天结算兜底) ---
    if state.hasTimedOut:
        if phase != 'OFF' and trip_start is not None:
            end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
            trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
            if trip_record: output_rows.append(trip_record)
            # 超时情况下无法获取最新行坐标，默认用 0.0 或由上一次 last_ts 逻辑缓存的值
            flush_all_ongoing_events(end_time_fallback, 0.0, 0.0)
        
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
            
            # 动态获取当前行 GPS 坐标 (请替换为实际业务中的 GPS 经纬度字段名)
            lon = float(row.get('gps_longitude', 0.0)) if pd.notna(row.get('gps_longitude')) else 0.0
            lat = float(row.get('gps_latitude', 0.0)) if pd.notna(row.get('gps_latitude')) else 0.0
            
            # [时间异常裂断防御] 
            if phase != 'OFF' and last_ts is not None:
                if (ts - last_ts).total_seconds() > 900: 
                    end_time_fallback = hang_start if phase == 'HANG_OFF' else last_ts
                    trip_record = emit_trip(trip_type_so_far, trip_start, end_time_fallback, trip_speeds)
                    if trip_record: output_rows.append(trip_record)
                    flush_all_ongoing_events(end_time_fallback, lon, lat)
                    
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
                    flush_all_ongoing_events(hang_start, lon, lat)
                    
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
                        # 开始GPS为行程启动点，结束GPS为第一次车速>0的点
                        evt = emit_sub_event('PARK_BEFORE_RUN_EVENT', trip_start, ts, lon, lat, lon, lat)
                        if evt: output_rows.append(evt)
                        first_speed_emitted = True
                        
                # === 注意：原先这里的 if power == 0: 下电判定已被移动到最下方 ===

            # ================= 二级事件检测引擎 (全面覆盖 23 个事件) =================
            if phase == 'ACTIVE':
                # 构造当前时刻全部二级事件的布尔状态映射表 (字段名请映射到实际数仓DWD层字段)
                current_conditions = {
                    # A. 车速区间事件 (持续时间要求见第二列参数：300秒或0秒)
                    'CREEP_RUN_EVENT': (0 < speed <= 10, 300),
                    'TRAFFIC_JAM_EVENT': (10 < speed <= 20, 300),
                    'LOW_SPEED_EVENT': (0 < speed <= 40, 0),
                    'MEDIUM_SPEED_EVENT': (40 < speed <= 80, 0),
                    'HIGH_SPEED_EVENT': (80 < speed <= 120, 0),
                    'OVER_SPEED_EVENT': (speed > 120, 0),
                    'PARK_RUNNING_EVENT': (speed == 0, 0),
                    
                    # B. 车机硬件与开关事件 (持续时间要求 1 分钟 = 60 秒)
                    'ON_OFF_IVI_EVENT': (row.get('ivi_power_status', 0) == 1, 0), # 车机上电状态
                    'TRUNK_EVENT': (row.get('trunk_door_status', 0) == 1, 60),
                    
                    # C. 座椅通风系列事件 (持续 >= 60秒)
                    'DRIV_SEAT_VENTNSTS_EVENT': (row.get('driv_seat_vent_sts', 0) == 1, 60),
                    'PASS_SEAT_VENTNSTS_EVENT': (row.get('pass_seat_vent_sts', 0) == 1, 60),
                    'SECRL_SEAT_VENTNSTS_EVENT': (row.get('secrl_seat_vent_sts', 0) == 1, 60),
                    'SECRR_SEAT_VENTNSTS_EVENT': (row.get('secrr_seat_vent_sts', 0) == 1, 60),
                    
                    # D. 座椅加热系列事件 (持续 >= 60秒)
                    'DRIV_SEAT_HEATSTS_EVENT': (row.get('driv_seat_heat_sts', 0) == 1, 60),
                    'PASS_SEAT_HEATSTS_EVENT': (row.get('pass_seat_heat_sts', 0) == 1, 60),
                    'SECRL_SEAT_HEATSTS_EVENT': (row.get('secrl_seat_heat_sts', 0) == 1, 60),
                    'SECRR_SEAT_HEATSTS_EVENT': (row.get('secrr_seat_heat_sts', 0) == 1, 60),
                    
                    # E. 座椅按摩系列事件 (持续 >= 60秒)
                    'DRIV_SEAT_MASSG_EVENT': (row.get('driv_seat_massg_sts', 0) == 1, 60),
                    'PASS_SEAT_MASSG_EVENT': (row.get('pass_seat_massg_sts', 0) == 1, 60),
                    'SECRL_SEAT_MASSG_EVENT': (row.get('secrl_seat_massg_sts', 0) == 1, 60),
                    'SECRR_SEAT_MASSG_EVENT': (row.get('secrr_seat_massg_sts', 0) == 1, 60)
                }

                # 核心高阶双端匹配逻辑
                for evt_name, (is_active, min_dur_sec) in current_conditions.items():
                    if is_active:
                        if evt_name not in ongoing_events:
                            # 触发进入事件：动态打包时间、GPS经纬度作为开始节点要素
                            ongoing_events[evt_name] = {
                                "start_time": str(ts),
                                "start_lon": lon,
                                "start_lat": lat
                            }
                    else:
                        if evt_name in ongoing_events:
                            # 触发退出事件：取出开始要素，结合当前行(结束节点)结算
                            evt_ctx = ongoing_events.pop(evt_name)
                            start_ts_pd = pd.to_datetime(evt_ctx["start_time"])
                            s_lon = evt_ctx["start_lon"]
                            s_lat = evt_ctx["start_lat"]
                            
                            duration = (ts - start_ts_pd).total_seconds()
                            if duration >= min_dur_sec:
                                evt = emit_sub_event(evt_name, start_ts_pd, ts, s_lon, s_lat, lon, lat)
                                if evt: output_rows.append(evt)

            # === [核心修复] 将下电挂起判定移到最后执行，确保最后时刻的开关状态能被上面的循环捕获闭合 ===
            if phase == 'ACTIVE' and power == 0:
                phase = 'HANG_OFF'
                hang_start = ts
            # =========================================================================================

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
        trigger_conf = {"availableNow": True} # 核心：单次读取全部历史片段后安全退出
        
    elif run_mode == "realtime":
        print(">>> [流处理] 启动实时流模式，绑定 Kafka DWD 实时主题...")
        raw_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "10.0.0.1:9092,10.0.0.2:9092") \
            .option("subscribe", "dwd_vehicle_logs_topic") \
            .option("startingOffsets", "latest") \
            .load() \
            .selectExpr("CAST(value AS STRING) as json_str")
        # 此处需通过 from_json 将 json_str 展平为对应的字段
        trigger_conf = {"processingTime": "10 seconds"}

    # 接入状态机核心算子
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

    # 批量汇聚写入 ByteHouse / ClickHouse
    def write_to_bytehouse(batch_df, batch_id):
        batch_df.persist() 
        valid_rows_df = batch_df.filter(F.col("row_type").isin('TRIP', 'EVENT'))
        
        if not valid_rows_df.isEmpty():
            print(f">>> 批次 {batch_id} 解析成功，正在增量下沉 {valid_rows_df.count()} 条记录至数仓 ADS 层...")
            valid_rows_df.write \
                .format("jdbc") \
                .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
                .option("url", "jdbc:bytehouse://your-bytehouse-endpoint:8123") \
                .option("dbtable", "ads_telematics.ads_trip_event_summary") \
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