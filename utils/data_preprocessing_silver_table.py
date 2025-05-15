import pyspark
import os
import re

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, split, explode, trim, lower, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

def parse_type_of_loan(value : str):
    """
    Parse the type of loan that was used
    """
    # Remove all spaces, instances of the word "and" and "loan"
    value = value.replace(" ", "").replace("and", "").replace("loan", "")

    return value

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

def parse_payment_behaviour(value : str):
    if value is None:
        return None
    if "payment" not in str(value):
        return "unknown"
    else:
        return lower(str(value))
        

def parse_credit_mix(value : str):
    if value == "Bad":
        return 0
    elif value == "Good":
        return 2
    elif value == "Standard":
        return 1
    else:
        # No credit
        return -1  # Other categories

def parse_credit_history_age(value : str):
    """
    Value is in years and months, convert to float of years
    """
    # Use regex to extract years and months
    years_match = re.search(r'(\d+)\s*Years?', value)
    months_match = re.search(r'(\d+)\s*Months?', value)

    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    
    # Convert months into fraction of a year and add to years
    return years + (months / 12.0)

def parse_payment_min_amount(value : str):
    """
    Yes or no, change to 1 or 0
    """
    if value == "Yes":
        return True
    else:
        # There is an additional column data called "NM", which may mean "Not Made"
        return False

def parse_changed_credit_limit(value : str):
    """
    Some Null values, just change to 0
    """
    try:
        cur_value = float(value)
        return cur_value
    except:
        return 0.0
    
def parse_occupation(value : str):
    """
    Unknown occupation, parse all empty strings
    """
    if "_" in value:
        return "Unknown"
    else:
        return value


def get_true_number_of_loan(loan_type : str):
    number_of_loans = len(loan_type.split(","))

    return number_of_loans

# Parsing functions
parse_type_of_loan_udf = F.udf(parse_type_of_loan, StringType())
parse_int_udf = F.udf(parse_int, IntegerType())
parse_float_udf = F.udf(parse_float, FloatType())
parse_credit_mix_udf = F.udf(parse_credit_mix, IntegerType())
parse_credit_history_age_udf = F.udf(parse_credit_history_age, FloatType())
parse_payment_min_amount_udf = F.udf(parse_payment_min_amount, BooleanType())
parse_changed_credit_limit_udf = F.udf(parse_changed_credit_limit, FloatType())
parse_occupation_udf = F.udf(parse_occupation, StringType())
parse_payment_behaviour_udf = F.udf(parse_payment_behaviour, StringType())

# Repair functions
get_true_number_of_loan_udf = F.udf(get_true_number_of_loan, IntegerType())

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
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

def process_silver_table_feature_financials(bronze_feature_financials : str, silver_table_dir : str, date : str, spark : SparkSession):
    """
    Process Feature finanicals
    """
    df = spark.read.csv(bronze_feature_financials, header=True, inferSchema=True)
    print(f"Loaded {bronze_feature_financials}, row count {df.count()}")

    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(), # Has weird values, need to handle 
        "Type_of_Loan": StringType(),  # List-like string, can parse later
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),  # e.g. "Bad", "Good", etc.
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": FloatType(),  # Date is in Year and months, need to parse
        "Payment_of_Min_Amount": StringType(),  # "Yes"/"No"
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),  # Categorical, need more information to determine whether to remain categorical or ordinal
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        if column == "Type_of_Loan":
            df = df.withColumn(column, parse_type_of_loan_udf(col(column)).cast(new_type))
        elif column == "Credit_Mix":
            df = df.withColumn(column, parse_credit_mix_udf(col(column)).cast(new_type))
        elif column == "Credit_History_Age":
            df = df.withColumn(column, parse_credit_history_age_udf(col(column)).cast(new_type))
        elif column == "Payment_of_Min_Amount":
            df = df.withColumn(column, parse_payment_min_amount_udf(col(column)).cast(new_type))
        elif column == "Changed_Credit_Limit":
            df = df.withColumn(column, parse_changed_credit_limit_udf(col(column)).cast(new_type))
        # Need to fix the values
        elif new_type == IntegerType():
            df = df.withColumn(column, parse_int_udf(col(column)).cast(new_type))
        elif new_type == FloatType():
            df = df.withColumn(column, parse_float_udf(col(column)).cast(new_type))
        else:
            df = df.withColumn(column, col(column).cast(new_type))

    # We can do some repair on the number of loans by checking the values from the loan types
    # and replacing the value inside depending on the number of loans
    df = df.withColumn("Num_of_Loan", get_true_number_of_loan_udf(col("Type_of_Loan")).cast(IntegerType()))

    # We need to remove any anomalous values within the dataset
    # Preseeded some None values so we can remove that first
    df = df.dropna()

    # Remove anomalous data based on certain values
    # e.g. > 50 credit cards, more than 50 loans, CUS_0x1140 where monthly salary 914 but yearly salary is 14 millions
    # df = df.filter(col("Outstanding_Debt") >= 0)
    print("---------Filtering Anomalous Data------------")
    df = df.filter(col("Annual_Income") >= 0)
    df = df.filter(col("Monthly_Inhand_Salary") >= 0)
    df = df.filter((col("Num_Bank_Accounts") <= 20) & (col("Num_Bank_Accounts") > 0))
    df = df.filter((col("Num_Credit_Card") <= 20) & (col("Num_Credit_Card") >= 0))
    df = df.filter((col("Num_of_Loan") <= 20) & (col("Num_of_Loan") >= 0))
    df = df.filter((col("Interest_Rate") <= 600) & (col("Interest_Rate") >= 0)) # According to online, max rating is ~600%
    df = df.filter((col("Monthly_Inhand_Salary") * 24 < col("Annual_Income")))

    # Remove filter columns, then change loan type to counter column
    print("---------Extracting Loan Data------------")
    # df_split = df.withColumn("loan_type_array", split("Type_of_Loan", ","))
    # df_exploded = df_split.withColumn("loan_type", explode("loan_type_array"))
    # df_normalized = df_exploded.withColumn("loan_type", trim(lower("loan_type")))
    # df_normalized.show(10)
    # loan_types = [row.loan_type for row in df_normalized.select("loan_type").distinct().collect()]

    # for loan in loan_types:
    #     df = df.withColumn(
    #         loan,
    #         when(lower(col("Type_of_Loan")).contains(loan), 1).otherwise(0)
    #     )
    
    # We then can count the number of each time of loan based off the Num_of_loan column

    # save silver table - IRL connect to database to write
    print("---------Writing Pandas File------------")
    df.show(5)
    partition_name = "silver_feature_finanicals_" + date + '.csv'
    filepath = os.path.join(silver_table_dir, partition_name)

    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    
    return df

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

    df.toPandas().to_csv(filepath)
    print('saved to:', filepath)

def process_silver_table_features_attributes(bronze_feature_attributes : str, silver_table_dir : str, date : str, spark : SparkSession):
    """
    Process Feature Attributes
    """
    df = spark.read.csv(bronze_feature_attributes, header=True, inferSchema=True)
    print(f"Loaded {bronze_feature_attributes}, row count {df.count()}")

    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        if new_type == IntegerType():
            df = df.withColumn(column, parse_int_udf(col(column)).cast(new_type))
        if column == "Occupation":
            df = df.withColumn(column, parse_occupation_udf(col(column)).cast(new_type))
        else:
            df = df.withColumn(column, col(column).cast(new_type))
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_attributes_" + date + '.csv'
    filepath = os.path.join(silver_table_dir, partition_name)

    df.toPandas().to_csv(filepath)
    print('saved to:', filepath)