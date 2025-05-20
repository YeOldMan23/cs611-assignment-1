import os
import shutil
import argparse
from datetime import datetime
from tqdm import tqdm

import pyspark
from pyspark.sql import SparkSession

from utils.data_preprocessing_bronze_table import *
from utils.data_preprocessing_silver_table import *
from utils.data_preprocessing_gold_table import *
from utils.helper import *

current_directory = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "cs611-assignment-1")))
csv_dir = os.path.join(current_directory, "data")
print(current_directory)

# Other parts
data_mart_dir = os.path.join(current_directory, "datamart")
bronze_dir = os.path.join(data_mart_dir, "bronze")
silver_dir = os.path.join(data_mart_dir, "silver")
gold_dir = os.path.join(data_mart_dir, "gold")

"""
Bronze Feature Function
"""
def data_prep_bronze(start_date, end_date, spark : SparkSession):
    print('\n\n---starting Bronze Table job---\n\n')

    # Get all the datetimes 
    dates_str_list = generate_first_of_month_dates(start_date, end_date)

    # We can build the bronze table
    # Get csvs
    csv_files = os.listdir(csv_dir)

    for csv_file in csv_files:
        csv_full_dir = os.path.join(csv_dir, csv_file)
        for date_str in dates_str_list:
            print("Preparing bronze table {}".format(csv_file))
            prepare_bronze_table_daily(csv_full_dir, bronze_dir, spark, date_str)

"""
Silver Feature Function
"""

def data_prep_silver(start_date, end_date, spark : SparkSession):
    print('\n\n---starting Silver table job---\n\n')

    # Get all the datetimes 
    dates_str_list = generate_first_of_month_dates(start_date, end_date)

    # We can build the silver table
    for date_str in dates_str_list:
        # Build the silver table for each csv
        expected_lms_loan_daily_file_name = "bronze_lms_loan_daily_" + date_str + ".csv"
        expected_loan_full_dir = os.path.join(bronze_dir, expected_lms_loan_daily_file_name)

        process_silver_table_loan_daily(expected_loan_full_dir,
                                        silver_dir,
                                        date_str,
                                        spark)
        
        expected_feature_financials_file_name = "bronze_features_financial_" + date_str + ".csv"
        expected_financial_full_dir = os.path.join(bronze_dir, expected_feature_financials_file_name)

        process_silver_table_feature_financials(expected_financial_full_dir,
                                                silver_dir,
                                                date_str,
                                                spark)
        
        expected_feature_attributes_file_name = "bronze_features_attribute_" + date_str + ".csv"
        expected_feature_attributes_full_dir = os.path.join(bronze_dir, expected_feature_attributes_file_name)

        process_silver_table_features_attributes(expected_feature_attributes_full_dir,
                                                 silver_dir,
                                                 date_str,
                                                 spark)
        
        expected_feature_clickstream_file_name = "bronze_feature_clickstream_" + date_str + ".csv"
        expected_feature_clickstream_full_dir = os.path.join(bronze_dir, expected_feature_clickstream_file_name)

        process_silver_table_features_clickstream(expected_feature_clickstream_full_dir,
                                                  silver_dir,
                                                  date_str,
                                                  spark)
        

"""
Gold Feature Function
"""

def data_prep_gold(start_date, end_date, spark : SparkSession):
    print('\n\n---starting Gold table job---\n\n')

    dates_str_list = generate_first_of_month_dates(start_date, end_date)

    label_df = None
    features_df = None

    # We can build the silver table
    for date_str in tqdm(dates_str_list):
        # Prepare the gold labels
        if label_df:
            cur_label_df = process_labels_gold_table(date_str, silver_dir, gold_dir, spark, dpd = 60, mob = 7)
            label_df = label_df.unionByName(cur_label_df)
        else:
            label_df = process_labels_gold_table(date_str, silver_dir, gold_dir, spark, dpd = 60, mob = 7)

        # Prepare the gold features
        if features_df:
            cur_feature_df = process_features_gold_table(date_str, silver_dir, gold_dir, spark)
            features_df = features_df.unionByName(cur_feature_df)
        else:
            features_df = process_features_gold_table(date_str, silver_dir, gold_dir, spark)

    # Save the data
    current_date = datetime.now().strftime("%Y-%m-%d")

    label_name = f"snapdate_{current_date}_" + "gold_label_store" + start_date + "to" + end_date + ".csv"
    label_filepath = os.path.join(gold_dir, label_name)
    label_df.toPandas().to_csv(label_filepath, index=False)
    print('labels saved to : ', label_filepath, " row count : ", label_df.count())

    feature_name = f"snapdate_{current_date}_" + "gold_feature_store" + start_date + "to" + end_date + ".csv"
    feature_filepath = os.path.join(gold_dir, feature_name)
    features_df.toPandas().to_csv(feature_filepath, index=False)
    print(f"saved to : {feature_filepath}, row count : {features_df.count()}")


    return label_df, features_df



if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", default="2023-01-01", type=str, help="YYYY-MM-DD")
    parser.add_argument("--start_date", default="2023-01-01", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end_date", default="2024-12-01", type=str, help="YYYY-MM-DD")

    args = parser.parse_args()

    print(f"Making Datamart, Start Date {args.start_date} End Date {args.end_date}")

    # Build File directories first
    current_directory = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "cs611-assignment-1")))
    csv_dir = os.path.join(current_directory, "data")
    print(current_directory)

    # Other parts
    data_mart_dir = os.path.join(current_directory, "datamart")
    bronze_dir = os.path.join(data_mart_dir, "bronze")
    silver_dir = os.path.join(data_mart_dir, "silver")
    gold_dir = os.path.join(data_mart_dir, "gold")

    # Refresh current directory
    if os.path.exists(data_mart_dir):
        shutil.rmtree(data_mart_dir)

    os.mkdir(data_mart_dir)
    os.mkdir(bronze_dir)
    os.mkdir(silver_dir)
    os.mkdir(gold_dir)
    
    # Call main with arguments explicitly passed
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # Run the entire chan
    data_prep_bronze(args.start_date, args.end_date, spark)
    data_prep_silver(args.start_date, args.end_date, spark)
    data_prep_gold(args.start_date, args.end_date, spark)

    print("---------JOB COMPLETE---------")