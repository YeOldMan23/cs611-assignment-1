import pyspark
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def prepare_bronze_table_daily(file_dir : str, 
                                    bronze_dir : str, 
                                    spark : SparkSession, 
                                    date : str,
                                    csv_type : str):
    """
    Loan Daily Bronze Table Prep
    Need to adjust by type
    """
    # Open the csv file path from the given input
    df = spark.read.csv(file_dir, header=True, inferSchema=True).filter(col('snapshot_date') == date)
    print(f"Row Count for Date {date} : {df.count()}")

    # Save data to datamart
    file_dir = f"bronze_{csv_type}_daily_" + date.replace('-', "_") + ".csv"
    global_file_dir = os.path.join(bronze_dir, file_dir)
    df.toPandas().to_csv(global_file_dir, index=False)
    print(f"Bronze {csv_type} Daily Date {date} saved to : {global_file_dir}")

    return df
