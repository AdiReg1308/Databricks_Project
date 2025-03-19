# Databricks notebook source
# MAGIC %pip install Pdfplumber langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from pyspark.sql.functions import substring_index

directory_path = "/Volumes/genai-poc-catalog/rag/pdf_vol"
file_paths = [file.path for file in dbutils.fs.ls(directory_path)]
df = spark.createDataFrame(file_paths, "string").select(substring_index("value", "/", -1).alias("file_name"))

# COMMAND ----------

df.show()

# COMMAND ----------

import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_volume_path = "/Volumes/genai-poc-catalog/rag/pdf_vol"
processed_files = spark.sql(f"SELECT DISTINCT file_name FROM `genai-poc-catalog`.rag.docs_track").collect()
processed_files = set(row["file_name"] for row in processed_files)

# COMMAND ----------

new_files = [file for file in os.listdir(pdf_volume_path) if file not in processed_files]
processed_files

# COMMAND ----------

all_text = ''  # Initailization 

for file_name in new_files:
    pdf_path = os.path.join(pdf_volume_path, file_name)

    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in pdf.pages:
            single_page_text = pdf_page.extract_text()
            all_text = all_text + '\n' + single_page_text

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

length_function = len

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=length_function,
)
chunks = splitter.split_text(all_text)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType
import pandas as pd

@pandas_udf("array<string>")
def get_chunks(dummy):
    return pd.Series([chunks])

spark.udf.register("get_chunks_udf", get_chunks)

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into `genai-poc-catalog`.rag.docs_text (text)
# MAGIC select explode(get_chunks_udf('dummy')) as text;

# COMMAND ----------

df.createOrReplaceTempView("temp_table") 
spark.sql("""
    INSERT INTO `genai-poc-catalog`.rag.docs_track
    SELECT * FROM temp_table
    WHERE NOT EXISTS (
        SELECT 1 FROM `genai-poc-catalog`.rag.docs_track
        WHERE temp_table.file_name = `genai-poc-catalog`.rag.docs_track.file_name
    )
""")

# COMMAND ----------

