import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# 导入你写好的核心算子和 Schema 
from src.jobs.trip_unified_job import process_unified_trip_state, RESULT_SCHEMA, STATE_SCHEMA

def generate_mock_can_data(output_dir: str):
    """
    生成 100 辆车的测试数据，30秒一条。
    包含特定的边界场景注入。
    """
    records = []
    base_time = datetime(2026, 5, 30, 8, 0, 0)
    
    # ==========================================
    # 场景 1: VIN_001 (完美行程与事件触发)
    # 预期: 触发 CREEP_RUN_EVENT, LOW_SPEED_EVENT, TRUNK_EVENT
    # ==========================================
    curr_time = base_time
    for i in range(20): # 10分钟 (20 * 30s)
        # 前 6 分钟 (12条): 速度 5 (蠕行)，后备箱关闭
        # 后 4 分钟 (8条) : 速度 0 (驻车)，后备箱打开
        speed = 5.0 if i < 12 else 0.0
        trunk = 0 if i < 12 else 1
        records.append({
            "vin": "VIN_001", "timestamp": curr_time, "syspowermod_2012001": 2,
            "vehspd_2011002": speed, "trunk_door_status": trunk,
            "gps_longitude": 120.0, "gps_latitude": 30.0
        })
        curr_time += timedelta(seconds=30)
    # 下电
    records.append({"vin": "VIN_001", "timestamp": curr_time, "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # ==========================================
    # 场景 2: VIN_002 (边界未满过滤测试)
    # 预期: 【不应】触发 CREEP_RUN_EVENT (未满5分), 【不应】触发 热座椅事件 (未满1分)
    # ==========================================
    curr_time = base_time
    for i in range(11): # 5.5分钟 (11 * 30s)
        # 前 4.5 分钟 (9条, 270秒): 速度 5 (不够5分钟蠕行阈值)
        # 然后速度突变到 45，打断蠕行状态
        speed = 5.0 if i < 9 else 45.0
        # 座椅加热只开 30秒 (1条)
        heat = 1 if i == 5 else 0
        records.append({
            "vin": "VIN_002", "timestamp": curr_time, "syspowermod_2012001": 2,
            "vehspd_2011002": speed, "driv_seat_heat_sts": heat,
            "gps_longitude": 120.0, "gps_latitude": 30.0
        })
        curr_time += timedelta(seconds=30)
    records.append({"vin": "VIN_002", "timestamp": curr_time, "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # ==========================================
    # 场景 3: VIN_003 (时间断层跳跃测试)
    # 预期: 生成 2 段独立的 TRIP
    # ==========================================
    curr_time = base_time
    # 第一段行驶
    for i in range(5):
        records.append({"vin": "VIN_003", "timestamp": curr_time, "syspowermod_2012001": 2, "vehspd_2011002": 20.0})
        curr_time += timedelta(seconds=30)
    # 突然跳跃 30 分钟 (模拟断网/脏数据)
    curr_time += timedelta(minutes=30)
    # 第二段行驶
    for i in range(5):
        records.append({"vin": "VIN_003", "timestamp": curr_time, "syspowermod_2012001": 2, "vehspd_2011002": 30.0})
        curr_time += timedelta(seconds=30)
    records.append({"vin": "VIN_003", "timestamp": curr_time, "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # ==========================================
    # 场景 4: 噪音背景车 (VIN_005 ~ VIN_100)
    # 生成一天内的随机启停，测试 OOM 和基本逻辑
    # ==========================================
    for v in range(5, 101):
        vin = f"VIN_{v:03d}"
        curr_time = base_time
        # 每辆车随便跑 10 圈 (5分钟)
        for i in range(10):
            records.append({
                "vin": vin, "timestamp": curr_time, "syspowermod_2012001": 2,
                "vehspd_2011002": float(np.random.randint(0, 60))
            })
            curr_time += timedelta(seconds=30)
        records.append({"vin": vin, "timestamp": curr_time, "syspowermod_2012001": 0, "vehspd_2011002": 0.0})

    # =======================================================================
    # 👇 [核心修复 v2] 场景 5: 应用层强行断层闭合 (Application-Level Flush)
    # 给所有生成过的车辆，强行追加 2 小时后的离线数据。逼出所有滞留内存。
    # =======================================================================
    future_time = base_time + timedelta(hours=2)
    all_vins = ["VIN_001", "VIN_002", "VIN_003"] + [f"VIN_{v:03d}" for v in range(5, 101)]
    for test_vin in all_vins:
        records.append({
            "vin": test_vin, "timestamp": future_time, "syspowermod_2012001": 0,
            "vehspd_2011002": 0.0, "trunk_door_status": 0, "driv_seat_heat_sts": 0,
            "gps_longitude": 0.0, "gps_latitude": 0.0
        })

    # 将数据转为 DataFrame 并写入临时目录 (Parquet格式)
    df = pd.DataFrame(records)
    
    # 填充 NaN 以防缺失键报错
    df = df.fillna({"trunk_door_status": 0, "driv_seat_heat_sts": 0, "gps_longitude": 0.0, "gps_latitude": 0.0})
    
    # 强制将 Pandas 默认的 64 位整数转换为 Spark 期望的 32 位整数
    df = df.astype({
        "syspowermod_2012001": "int32", 
        "trunk_door_status": "int32", 
        "driv_seat_heat_sts": "int32"
    })
    
    parquet_path = os.path.join(output_dir, "mock_data.parquet")
    df.to_parquet(parquet_path, index=False)
    return parquet_path

def run_test_pipeline():
    print(">>> 1. 初始化本地 Spark Session...")
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("TripEngineUnitTest") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        checkpoint_dir = os.path.join(temp_dir, "checkpoint")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        
        print(">>> 2. 生成 100 辆车，30s 间隔的模拟数据...")
        generate_mock_can_data(input_dir)
        
        print(">>> 3. 运行流式处理算子 (模拟 AvailableNow 批流一体)...")
        # 读取 Mock 数据
        raw_stream = spark.readStream.schema(
            "vin STRING, timestamp TIMESTAMP, syspowermod_2012001 INT, vehspd_2011002 DOUBLE, trunk_door_status INT, driv_seat_heat_sts INT, gps_longitude DOUBLE, gps_latitude DOUBLE"
        ).parquet(input_dir)

        # 核心算子
        enriched_stream = raw_stream \
            .withWatermark("timestamp", "10 minutes") \
            .groupBy("vin") \
            .applyInPandasWithState(
                func=process_unified_trip_state,
                outputStructType=RESULT_SCHEMA,
                stateStructType=STATE_SCHEMA,
                outputMode="append",
                timeoutConf="EventTimeTimeout"
            )

        # # 写入临时内存表进行断言
        # query = enriched_stream.writeStream \
        #     .format("memory") \
        #     .queryName("test_results") \
        #     .outputMode("append") \
        #     .option("checkpointLocation", checkpoint_dir) \
        #     .trigger(availableNow=True) \
        #     .start()

        # query.awaitTermination()

        # ====== 替换原来的 memory sink，改为 csv 落盘 sink ======
        output_csv_dir = "./test_output_data" # 结果会保存在你项目根目录的这个文件夹下
        
        query = enriched_stream.coalesce(1).writeStream \
            .format("csv") \
            .option("header", "true") \
            .option("path", output_csv_dir) \
            .option("checkpointLocation", checkpoint_dir) \
            .outputMode("append") \
            .trigger(availableNow=True) \
            .start()

        query.awaitTermination()
        print(f">>> 🎉 完整明细数据已成功落盘！请至目录查看: {os.path.abspath(output_csv_dir)}")
        # =======================================================


        print(">>> 4. 开始执行断言验证 (Assertions)...")
        output_csv_dir = "./test_output_data"
        results_df = spark.read.option("header", "true").csv(output_csv_dir).toPandas()

        # ---------- 断言 1: VIN_001 事件触发 ----------
        vin1_events = results_df[(results_df['vin'] == 'VIN_001') & (results_df['row_type'] == 'EVENT')]
        event_names = vin1_events['trip_type'].tolist()
        assert 'CREEP_RUN_EVENT' in event_names, "VIN_001 应该触发蠕行事件"
        assert 'TRUNK_EVENT' in event_names, "VIN_001 应该触发后备箱事件"
        print("✅ 断言 1 通过: VIN_001 成功触发了符合时长的持续事件。")

        # ---------- 断言 2: VIN_002 边界过滤 ----------
        vin2_events = results_df[(results_df['vin'] == 'VIN_002') & (results_df['row_type'] == 'EVENT')]
        event_names_2 = vin2_events['trip_type'].tolist()
        assert 'CREEP_RUN_EVENT' not in event_names_2, "VIN_002 不应触发蠕行事件 (仅 270 秒)"
        assert 'DRIV_SEAT_HEATSTS_EVENT' not in event_names_2, "VIN_002 不应触发加热事件 (仅 30 秒)"
        print("✅ 断言 2 通过: VIN_002 的短时噪音数据被成功过滤。")

        # ---------- 断言 3: VIN_003 异常跳跃切断 ----------
        vin3_trips = results_df[(results_df['vin'] == 'VIN_003') & (results_df['row_type'] == 'TRIP')]
        assert len(vin3_trips) == 2, f"VIN_003 应该被切分成 2 段行程，实际得到 {len(vin3_trips)} 段"
        print("✅ 断言 3 通过: VIN_003 的 30 分钟断层数据成功切断了行程。")

        # ---------- 断言 4: 压测整体通过 ----------
        total_trips = results_df[results_df['row_type'] == 'TRIP'].shape[0]
        assert total_trips >= 100, "100 辆车至少应该有 100 趟行程"
        print(f"✅ 断言 4 通过: 100 辆车的压力测试完成，共输出 {total_trips} 趟行程记录。")

        print("\n🎉 所有测试均通过！状态机边界逻辑固若金汤。")

if __name__ == "__main__":
    run_test_pipeline()