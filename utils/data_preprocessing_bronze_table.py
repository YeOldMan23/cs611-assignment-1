import pyspark
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def prepare_bronze_table_daily(file_dir : str, 
                                    bronze_dir : str, 
                                    spark : SparkSession, 
                                    snapshotdate_str : str):
    """
    Loan Daily Bronze Table Prep
    Need to adjust by type
    """
    # Open the csv file path from the given input
    snapshotdate = datetime.strptime(snapshotdate_str, "%Y-%m-%d")
    
    df = spark.read.csv(file_dir, header=True, inferSchema=True).filter(col('snapshot_date') == snapshotdate)
    print(f"Row Count for Date {snapshotdate} : {df.count()}")

    # Lookup csv type
    csv_type = os.path.basename(file_dir).rstrip('.csv')

    # Save data to datamart
    file_dir = f"bronze_{csv_type}_" + snapshotdate_str + ".csv"
    global_file_dir = os.path.join(bronze_dir, file_dir)
    df.toPandas().to_csv(global_file_dir, index=False)
    print(f"Bronze {csv_type} Daily Date {snapshotdate} saved to : {global_file_dir}")

    return df
