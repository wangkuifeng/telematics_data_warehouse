import argparse
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType

# 引入 ETL 层的核心逻辑
from src.etl.trip_etl import transform_trip_data

def run_job(target_date: str, prev_date: str):
    """
    行程切分作业执行入口
    """
    # 1. 初始化 Spark Session
    spark = SparkSession.builder \
        .appName(f"Trip_Splitter_Job_{target_date}") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    try:
        # 2. 提取 T-1 日状态表数据
        try:
            state_df = spark.table("dwd_veh_trip_state").filter(F.col("dt") == prev_date)
        except Exception as e:
            print(f"Warning: Failed to load state for {prev_date}. Falling back to empty state. Reason: {e}")
            state_df = spark.createDataFrame([], schema=StructType([]))

        # 3. 提取 T 日新原始日志
        raw_df = spark.table("ods_veh_log").filter(F.col("dt") == target_date)

        # 4. 调用 ETL 层进行核心业务转换
        processed_df = transform_trip_data(state_df, raw_df)
        
        # 缓存结果，防止两次写操作触发重复计算
        processed_df.cache()

        # ==========================================
        # 5. 数据分流落表
        # ==========================================
        
        # A. 写入行程结果表
        trips_output = processed_df.filter(F.col("row_type") == 'TRIP').select(
            "vin", 
            "trip_type", 
            "start_time", 
            "end_time",
            "max_speed",
            "avg_speed"
        ).withColumn("dt", F.lit(target_date))
        
        trips_output.write \
            .mode("append") \
            .partitionBy("dt") \
            .insertInto("dwd_veh_trips_table")

        # B. 覆写 T 日状态表快照
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

        new_state_output.write \
            .mode("overwrite") \
            .partitionBy("dt") \
            .insertInto("dwd_veh_trip_state")

    finally:
        # 清理缓存并释放资源
        if 'processed_df' in locals():
            processed_df.unpersist()
        spark.stop()

if __name__ == "__main__":
    # 支持外部调度器 (如 DolphinScheduler / Airflow) 传入日期参数
    parser = argparse.ArgumentParser(description="Vehicle Trip Segmentation T+1 Job")
    parser.add_argument("--target_date", required=True, help="Target processing date (e.g., 2023-10-02)")
    parser.add_argument("--prev_date", required=True, help="Previous day state date (e.g., 2023-10-01)")
    
    args = parser.parse_args()
    
    print(f"Starting trip_job for target_date: {args.target_date}, with state from prev_date: {args.prev_date}")
    run_job(args.target_date, args.prev_date)