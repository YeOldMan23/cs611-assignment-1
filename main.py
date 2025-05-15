import os
import shutil
import argparse
from datetime import datetime

import pyspark
from pyspark.sql import SparkSession

from utils.data_preprocessing_bronze_table import *
from utils.data_preprocessing_silver_table import *
from utils.helper import *

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(current_directory, "data")

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

def data_prep(snapshotdate, start_date, end_date, spark : SparkSession):
    print('\n\n---starting job---\n\n')

    # Get all the datetimes 
    dates_str_list = generate_first_of_month_dates(start_date, end_date)

    # We can build the bronze table
    # Get csvs
    csv_files = os.listdir(csv_dir)

    for csv_file in csv_files:
        csv_full_dir = os.path.join(csv_dir, csv_file)
        for date_str in dates_str_list:
            prepare_bronze_table_daily(csv_full_dir, bronze_dir, spark, date_str)

    # Now we can prepare the silver
    for csv_file in csv_files:
        csv_full_dir = os.path.join(csv_dir, csv_file)
        csv_type = csv_file.rstrip(".csv")

        if csv_type == "lms_loan_daily":
            # Get all the lms_loan_daily files
            for date_str in dates_str_list:
                expected_lms_loan_daily_file_name = "bronze_" + csv_type + "_" + date_str + ".csv"
                expected_full_dir = os.path.join(bronze_dir, expected_lms_loan_daily_file_name)

                process_silver_table_loan_daily(expected_full_dir,
                                                silver_dir,
                                                date_str,
                                                spark)
                
        elif csv_type == "feature_finanicals":
            # Get all feature_financials files
            for date_str in dates_str_list:
                expected_feature_financials_file_name = "bronze_" + csv_type + "_" + date_str + ".csv"
                expected_full_dir = os.path.join(bronze_dir, expected_feature_financials_file_name)

                process_silver_table_feature_financials(expected_full_dir,
                                                        silver_dir,
                                                        date_str,
                                                        spark)


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", default="2023-01-01", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--start_date", default="2023-01-01", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", default="2024-12-01", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    data_prep(args.snapshotdate, args.start_date, args.end_date, spark)