import pyspark
import os
import re
import pandas as pd
import numpy as np
from collections import Counter

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, split, explode, trim, lower, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

def parse_int(value : str):
    """
    Remove the weird values from number of loan
    """
    if type(value) == int:
        return int(value)
    else:
        digit_match = re.search(r'\d+', value)
        if digit_match:
            return int(digit_match.group())
        else:
            return None
        
def parse_float(s):
    if s is None:
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
    if match:
        return float(match.group())
    return None

# Parsing functions
parse_int_udf = F.udf(parse_int, IntegerType())
parse_float_udf = F.udf(parse_float, FloatType())

"""
Silver Loan Daily
"""

def process_silver_table_loan_daily(bronze_loan_file : str, silver_table_dir : str, date : str, spark : SparkSession):
    """
    Clean the data here, differ based on the type of data that is taken 
    """
    # Get the corresponding data
    df = spark.read.csv(bronze_loan_file, header=True, inferSchema=True)
    print(f"Loaded {bronze_loan_file}, row count {df.count()}")

    # Clean the data
    column_type_map = {
        "loan_id" : StringType(),
        "Customer_ID" : StringType(),
        "loan_start_date" : DateType(),
        "tenure" : IntegerType(),
        "installment_num" : IntegerType(),
        "loan_amt" : FloatType(),
        "due_amt" : FloatType(),
        "paid_amt" : FloatType(),
        "overdue_amt" : FloatType(),
        "balance" : FloatType(),
        "snapshot_date" : DateType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_lms_loan_daily_" + date + '.csv'
    filepath = os.path.join(silver_table_dir, partition_name)
    print("Saving File {} row count {}".format(filepath, df.count()))

    # Convert to pandas and save as CSV
    df.toPandas().to_csv(filepath, index=False)

"""
Silver Feature Financials
"""

def extract_float(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"\d+(\.\d+)?", str(val).replace(",", ""))
    if match:
        return float(match.group())
    return np.nan

def extract_int(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"\d+", str(val).replace(",", ""))
    if match:
        return int(match.group())
    return np.nan

def count_loans(loan_str):
    if pd.isna(loan_str) or not isinstance(loan_str, str):
        return np.nan
    # Remove 'and' and split by comma
    loan_list = [x.strip().lower() for x in re.sub(r'\band\b', ',', loan_str, flags=re.IGNORECASE).split(',')]
    # Filter out empty strings and count
    loan_list = [loan for loan in loan_list if loan]
    return int(len(loan_list))

def count_distinct_loans(loan_str):
    def split_loans(loan_str):
        if pd.isna(loan_str):
            return []
        parts = re.sub(r"\band\b", ",", loan_str, flags=re.IGNORECASE).split(",")
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned
    loans = split_loans(loan_str)
    return Counter(loans)

def parse_credit_history_age(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(val))
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return round(years + months / 12, 2)
    # Fallback if only years or only months are present
    match_years = re.search(r"(\d+)\s*Years?", str(val))
    match_months = re.search(r"(\d+)\s*Months?", str(val))
    years = int(match_years.group(1)) if match_years else 0
    months = int(match_months.group(1)) if match_months else 0
    return round(years + months / 12, 2)

def process_silver_table_feature_financials(bronze_feature_financials : str, silver_table_dir : str, date : str, spark : SparkSession):
    """
    Process Feature finanicals,
    special case read as csv
    """
    all_loan_types = {'Not Specified', 
                      'Personal Loan', 
                      'Mortgage Loan', 
                      'Payday Loan', 
                      'Credit-Builder Loan', 
                      'Home Equity Loan',
                      'Debt Consolidation Loan', 
                      'Auto Loan', 
                      'Student Loan'}

    # Read the bronze file
    df = pd.read_csv(bronze_feature_financials, dtype=str)
    print(f"Loaded {bronze_feature_financials}, row count {len(df)}")

    # We then try to clean the data by trying to save some of the values
    df['Annual_Income'] = df["Annual_Income"].astype(str).apply(extract_float)
    df['Monthly_Inhand_Salary'] = df["Monthly_Inhand_Salary"].astype(str).apply(extract_float)
    df['Num_Bank_Accounts'] = df["Num_Bank_Accounts"].astype(str).apply(extract_int)
    df['Num_Credit_Card'] = df["Num_Credit_Card"].astype(str).apply(extract_int)
    df['Interest_Rate'] = df["Interest_Rate"].astype(str).apply(extract_float)
    df['Num_of_Loan'] = df["Num_of_Loan"].astype(str).apply(extract_int)
    df['Delay_from_due_date'] = df["Delay_from_due_date"].astype(str).apply(extract_int)
    df['Num_of_Delayed_Payment'] = df["Num_of_Delayed_Payment"].astype(str).apply(extract_int)
    df['Changed_Credit_Limit'] = df["Changed_Credit_Limit"].astype(str).apply(extract_float)
    df['Outstanding_Debt'] = df["Outstanding_Debt"].astype(str).apply(extract_float)
    df['Credit_Utilization_Ratio'] = df["Credit_Utilization_Ratio"].astype(str).apply(extract_float)
    df['Credit_History_Age'] = df["Credit_History_Age"].astype(str).apply(parse_credit_history_age)
    df['Total_EMI_per_month'] = df["Total_EMI_per_month"].astype(str).apply(extract_float)
    df['Amount_invested_monthly'] = df["Amount_invested_monthly"].astype(str).apply(extract_float)
    df['Monthly_Balance'] = df["Monthly_Balance"].astype(str).apply(extract_float)

    # We fix the empty values in credit mix
    df['Credit_Mix'] = df['Credit_Mix'].replace(to_replace=r'^_+$', value='Unknown', regex=True)


    # We can try to count the number of loans to try to save them
    df["Num_of_Loan"] = df["Type_of_Loan"].apply(count_loans).astype("Int64")
    df["Payment_Behaviour"] = df["Payment_Behaviour"].apply(
        lambda x: x if pd.notna(x) and "payment" in x.lower() else "Unknown"
    )
    
    # We can make new columns to turn boolean types into unique boolean variables
    for loan in all_loan_types:
        df[loan] = df["Type_of_Loan"].apply(lambda x: count_distinct_loans(x)[loan])

    # Drop any rows with NA values
    df = df.dropna()

    # We also filter anomalous data
    df = df[df["Annual_Income"] >= 0]
    df = df[df["Monthly_Inhand_Salary"] >= 0]
    df = df[(df["Num_Bank_Accounts"] <= 20) & (df["Num_Bank_Accounts"] > 0)]
    df = df[(df["Num_Credit_Card"] <= 20) & (df["Num_Credit_Card"] >= 0)]
    df = df[(df["Num_of_Loan"] <= 20) & (df["Num_of_Loan"] >= 0)]

    # Save the partitiion
    partition_name = "silver_feature_financials_" + date + ".csv"
    filepath = os.path.join(silver_table_dir, partition_name)
    print("Saving file : {} row count {}".format(filepath, len(df)))
    df.to_csv(filepath, index=False)
    
"""
Silver Feature Clickstream
"""

def process_silver_table_features_clickstream(bronze_feature_clickstream : str, silver_table_dir : str, date : str, spark : SparkSession):
    """
    Process Feature Clickstream
    """
    df = spark.read.csv(bronze_feature_clickstream, header=True, inferSchema=True)
    print(f"Loaded {bronze_feature_clickstream}, row count {df.count()}")

    column_type_map = {
        "fe_1": FloatType(),
        "fe_2": FloatType(),
        "fe_3": FloatType(),
        "fe_4": FloatType(),
        "fe_5": FloatType(),
        "fe_6": FloatType(),
        "fe_7": FloatType(),
        "fe_8": FloatType(),
        "fe_9": FloatType(),
        "fe_10": FloatType(),
        "fe_11": FloatType(),
        "fe_12": FloatType(),
        "fe_13": FloatType(),
        "fe_14": FloatType(),
        "fe_15": FloatType(),
        "fe_16": FloatType(),
        "fe_17": FloatType(),
        "fe_18": FloatType(),
        "fe_19": FloatType(),
        "fe_20": FloatType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_clickstream_" + date + '.csv'
    filepath = os.path.join(silver_table_dir, partition_name)

    print("Saving File {} row count {}".format(filepath, df.count()))

    # Convert PySpark DataFrame to pandas and save as CSV
    df.toPandas().to_csv(filepath, index=False)

"""
Silver Feature Attributes
"""

def process_silver_table_features_attributes(bronze_feature_attributes : str, silver_table_dir : str, date : str, spark : SparkSession):
    """
    Process Feature Attributes
    Dealing with String Again so need to cast to csv
    """
    df = pd.read_csv(bronze_feature_attributes, dtype=str)
    print(f"Loaded {bronze_feature_attributes}, row count {df.count()}")

    # Clean the numerical data
    df['Age'] = df["Age"].astype(str).apply(extract_int)

    # Clean occupation
    df.loc[df['Occupation'].str.fullmatch(r'_+'), 'Occupation'] = 'Unknown'
    
    # Drop Unknowns
    df = df.drop(columns=["SSN"])
    df = df.dropna()

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_attributes_" + date + '.csv'
    filepath = os.path.join(silver_table_dir, partition_name)
    print("Saving file : {} row count {}".format(filepath, len(df)))
    df.to_csv(filepath, index=False)
