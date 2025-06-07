import sys
import json
import boto3
import logging
import traceback
from datetime import datetime
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lit, year, month, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType

# Parameters
args = getResolvedOptions(
    sys.argv,
    [
        'SOURCE_BUCKET',
        'SOURCE_KEY',
        'EXECUTION_ID',
        'VALIDATION_RULES_PATH',
        'ERROR_PATH',
        'BAD_RECORDS_PATH',
        'TRANSFORMED_PATH',
        'ICEBERG_DB',
        'ICEBERG_TABLE',
        'CHECKPOINT_PATH',
        'RERUN_MODE', # 'Y' or 'N'
        'AUDIT_PATH'
    ]
)
SOURCE_BUCKET = args['SOURCE_BUCKET']
SOURCE_KEY = args['SOURCE_KEY']
EXECUTION_ID = args['EXECUTION_ID']
VALIDATION_RULES_PATH = args['VALIDATION_RULES_PATH']
ERROR_PATH = args['ERROR_PATH']
BAD_RECORDS_PATH = args['BAD_RECORDS_PATH']
TRANSFORMED_PATH = args['TRANSFORMED_PATH']
ICEBERG_DB = args['ICEBERG_DB']
ICEBERG_TABLE = args['ICEBERG_TABLE']
CHECKPOINT_PATH = args['CHECKPOINT_PATH']
RERUN_MODE = args['RERUN_MODE']
AUDIT_PATH = args['AUDIT_PATH']

# Set up Spark and Glue Context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

s3 = boto3.client('s3')
logger = logging.getLogger('GlueJob')
logger.setLevel(logging.INFO)

def write_audit(step, status, details=""):
    audit_record = {
        "execution_id": EXECUTION_ID,
        "timestamp": datetime.now().isoformat(),
        "source_file": SOURCE_KEY,
        "step": step,
        "status": status,
        "details": details
    }
    path = f"{AUDIT_PATH}/audit_{EXECUTION_ID}.json"
    s3.put_object(
        Bucket=SOURCE_BUCKET,
        Key=path,
        Body=json.dumps(audit_record) + "\n"
    )

def write_checkpoint(step):
    s3.put_object(
        Bucket=SOURCE_BUCKET,
        Key=f"{CHECKPOINT_PATH}/{EXECUTION_ID}_{step}.checkpoint",
        Body=b'COMPLETED'
    )

def checkpoint_exists(step):
    try:
        s3.head_object(Bucket=SOURCE_BUCKET, Key=f"{CHECKPOINT_PATH}/{EXECUTION_ID}_{step}.checkpoint")
        return True
    except Exception:
        return False

def read_validation_rules(path):
    bucket, key = path.replace("s3://", "").split("/", 1)
    rules_obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(rules_obj['Body'].read().decode('utf-8'))

def validate_data(df, rules):
    errors = []
    for rule in rules['rules']:
        colname = rule['column']
        condition = rule['condition']
        value = rule['value']
        if condition == "not_null":
            errs = df.filter(col(colname).isNull()).withColumn("error", lit(f"{colname} is null"))
            errors.append(errs)
        elif condition == "min":
            errs = df.filter(col(colname) < value).withColumn("error", lit(f"{colname} < {value}"))
            errors.append(errs)
        # Add more rule types as required
    if errors:
        bad_records = errors[0]
        for e in errors[1:]:
            bad_records = bad_records.union(e)
        good_records = df.join(bad_records.select("id"), "id", "left_anti")
    else:
        bad_records = spark.createDataFrame([], df.schema)
        good_records = df
    return good_records, bad_records

def move_file_to_error():
    copy_source = {'Bucket': SOURCE_BUCKET, 'Key': SOURCE_KEY}
    dest_key = f"{ERROR_PATH}/{SOURCE_KEY.split('/')[-1]}"
    s3.copy_object(CopySource=copy_source, Bucket=SOURCE_BUCKET, Key=dest_key)
    s3.delete_object(Bucket=SOURCE_BUCKET, Key=SOURCE_KEY)

def transformation(df):
    # Example: add column, rename, join, aggregate, filter
    df = df.withColumn("ingest_time", lit(datetime.now()))
    if "cust_id" in df.columns:
        df = df.withColumnRenamed("cust_id", "customer_id")
    # Dummy join example (requires a second DataFrame, skipped for brevity)
    # Aggregate: count by city
    agg_df = df.groupBy("city").count()
    # Filter example
    df = df.filter(col("status") == "active")
    # Partition columns
    df = df.withColumn("year", year(col("ingest_time"))) \
           .withColumn("month", month(col("ingest_time"))) \
           .withColumn("day", dayofmonth(col("ingest_time")))
    return df

def write_iceberg_table(df):
    output_path = f"s3://{SOURCE_BUCKET}/{TRANSFORMED_PATH}/"
    df.write.format("iceberg") \
        .mode("append") \
        .option("path", output_path) \
        .partitionBy("year", "month", "day") \
        .save(f"{ICEBERG_DB}.{ICEBERG_TABLE}")

# MAIN FLOW
try:
    # Checkpoint: Validation
    if RERUN_MODE == 'N' or not checkpoint_exists("validation"):
        write_audit("validation", "STARTED")
        # Read validation rules
        rules = read_validation_rules(VALIDATION_RULES_PATH)
        # Read CSV
        df = spark.read.option("header", True).csv(f"s3://{SOURCE_BUCKET}/{SOURCE_KEY}")
        # Validation
        good_df, bad_df = validate_data(df, rules)
        if bad_df.count() > 0:
            bad_path = f"s3://{SOURCE_BUCKET}/{BAD_RECORDS_PATH}/{EXECUTION_ID}_bad_records.csv"
            bad_df.write.mode("overwrite").csv(bad_path)
            move_file_to_error()
            write_audit("validation", "FAILED", "Validation failed. Bad records found.")
            sys.exit(1)
        write_checkpoint("validation")
        write_audit("validation", "SUCCESS")
    else:
        # Continue from Transformation
        df = spark.read.option("header", True).csv(f"s3://{SOURCE_BUCKET}/{SOURCE_KEY}")

    # Checkpoint: Transformation
    if RERUN_MODE == 'N' or not checkpoint_exists("transformation"):
        write_audit("transformation", "STARTED")
        transformed_df = transformation(df)
        # Save intermediate for potential reruns
        tmp_path = f"s3://{SOURCE_BUCKET}/{TRANSFORMED_PATH}/tmp/{EXECUTION_ID}/"
        transformed_df.write.mode("overwrite").parquet(tmp_path)
        write_checkpoint("transformation")
        write_audit("transformation", "SUCCESS")
    else:
        # Load from checkpoint
        tmp_path = f"s3://{SOURCE_BUCKET}/{TRANSFORMED_PATH}/tmp/{EXECUTION_ID}/"
        transformed_df = spark.read.parquet(tmp_path)

    # Checkpoint: Load
    if RERUN_MODE == 'N' or not checkpoint_exists("load"):
        write_audit("load", "STARTED")
        write_iceberg_table(transformed_df)
        write_checkpoint("load")
        write_audit("load", "SUCCESS")

except Exception as e:
    logger.error(traceback.format_exc())
    write_audit("job", "FAILED", str(e))
    sys.exit(1)

write_audit("job", "COMPLETED")
