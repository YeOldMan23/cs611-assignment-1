# Fuse the data to by year, the perform correlation analysis
import os
import pandas as pd
from datetime import datetime

import numpy as np
from .helper import generate_first_of_month_dates

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, greatest, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark : SparkSession, dpd, mob):
    # prepare arguments
    current_date = datetime.now().strftime("%Y-%m-%d")
    
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
    partition_name = f"snapdate_{current_date}_" + "gold_label_store_" + snapshot_date_str.replace('-','_') + '.csv'
    
    return df

def process_features_gold_table(snapshot_date_str, silver_dir, start_date, end_date, spark : SparkSession):
    current_date = datetime.now().strftime("%Y-%m-%d")

    # We need to access each of the stores from the specific dates
    silver_features_financials = os.path.join(silver_dir, "silver_feature_financials_" + snapshot_date_str + '.csv')
    silver_features_attributes = os.path.join(silver_dir, "silver_feature_attributes_" + snapshot_date_str + '.csv')

    ff_df = spark.read.csv(silver_features_financials, header=True, inferSchema=True)
    fa_df = spark.read.csv(silver_features_attributes, header=True, inferSchema=True)

    # Load the entire feature clickstream data, then filter by date
    dates_list = generate_first_of_month_dates(start_date, end_date)
    fc_df = None
    for date_str in dates_list:
        if fc_df:
            silver_features_clickstream = os.path.join(silver_dir, "silver_feature_clickstream_" + date_str + '.csv')
            cur_fc_df = spark.read.csv(silver_features_clickstream, header=True, inferSchema=True)
            fc_df = fc_df.unionByName(cur_fc_df)
        else:
            silver_features_clickstream = os.path.join(silver_dir, "silver_feature_clickstream_" + date_str + '.csv')
            fc_df = spark.read.csv(silver_features_clickstream, header=True, inferSchema=True)


    # fc_df = spark.read.csv(silver_features_clickstream, header=True, inferSchema=True)
    print('loaded from:', silver_features_financials, 'row count:', ff_df.count())
    print('loaded from:', silver_features_attributes, 'row count:', fa_df.count())
    print('loaded from:', silver_features_clickstream, 'row count:', fc_df.count())

    # Find the latest snapshot date afterwards
    # ff_df = ff_df.withColumnRenamed("snapshot_date", "snapshot_date_1")
    # ff_df = ff_df.withColumn("snapshot_date_1", to_date("snapshot_date_1"))

    fa_df = fa_df.withColumnRenamed("snapshot_date", "snapshot_date_2")
    fa_df = fa_df.withColumn("snapshot_date_2", to_date("snapshot_date_2"))

    # fc_df = fc_df.withColumnRenamed("snapshot_date", "snapshot_date_3")
    # fc_df = fc_df.withColumn("snapshot_date_3", to_date("snapshot_date_3"))

    # Merge the 3 datasets by date to correspond to the label store
    # Feature clickstream is the cleanest dataset, followed by attributes then finally the financials 
    df_joined_1 = ff_df.join(fa_df, on="Customer_ID", how="inner")
    final_df    = df_joined_1.join(fc_df, on=["Customer_ID", "snapshot_date"])
    
    # df_joined_1 = fc_df.join(fa_df, on="Customer_ID", how="inner")
    # final_df = df_joined_1.join(ff_df, on="Customer_ID", how="inner")

    # final_df = final_df.withColumn(
    #     "snapshot_date",
    #     greatest("snapshot_date_1", "snapshot_date_2", "snapshot_date_3")
    # )
    # final_df = final_df.drop("snapshot_date_1", "snapshot_date_2", "snapshot_date_3")
    final_df = final_df.drop("snapshot_date_2")

    print("Final Row Count : ", final_df.count())

    return final_df