# Fuse the data to by year, the perform correlation analysis
import os
import pandas as pd
from datetime import datetime

import numpy as np

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark : SparkSession, dpd, mob):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str + '.csv'
    filepath = os.path.join(silver_loan_daily_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(gold_label_store_directory, partition_name)
    
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_dir, gold_feature_store_directory, spark : SparkSession):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # We need to access each of the stores from the specific dates
    silver_features_financials = os.path.join(silver_dir, "silver_feature_financials_" + snapshot_date_str + '.csv')
    silver_features_attributes = os.path.join(silver_dir, "silver_feature_attributes_" + snapshot_date_str + '.csv')
    silver_features_clickstream = os.path.join(silver_dir, "silver_feature_clickstream_" + snapshot_date_str + '.csv')

    ff_df = spark.read.csv(silver_features_financials, header=True, inferSchema=True)
    fa_df = spark.read.csv(silver_features_attributes, header=True, inferSchema=True)
    fc_df = spark.read.csv(silver_features_clickstream, header=True, inferSchema=True)
    print('loaded from:', silver_features_financials, 'row count:', ff_df.count())
    print('loaded from:', silver_features_attributes, 'row count:', fa_df.count())
    print('loaded from:', silver_features_clickstream, 'row count:', fc_df.count())

    # Merge the 3 datasets by date to correspond to the label store
    # Feature clickstream is the cleanest dataset, followed by attributes then finally the financials 
    df_joined_1 = fc_df.join(fa_df, on="Customer_ID", how="inner")
    final_df = df_joined_1.join(ff_df, on="Customer_ID", how="inner")

    # Save the final file
    partition_name = "gold_feature_store_" + snapshot_date_str + '.csv'
    full_partition_path = os.path.join(gold_feature_store_directory, partition_name)
    final_df.toPandas().to_csv(full_partition_path, index=False)
    print(f"saved to : {full_partition_path}, row count : {final_df.count()}")

    return final_df