import os
from pyspark.sql.functions import substring_index
directory_path = "/Volumes/genai-poc-catalog/rag/pdf_vol"
file_paths = [file.path for file in dbutils.fs.ls(directory_path)]
df = spark.createDataFrame(file_paths, "string").select(substring_index("value", "/", -1).alias("file_name"))

df.show()

pdf_volume_path = "/Volumes/genai-poc-catalog/rag/pdf_vol"
processed_files = spark.sql(f"SELECT DISTINCT file_name FROM `genai-poc-catalog`.rag.docs_track").collect()
processed_files = set(row["file_name"] for row in processed_files)

new_files = [file for file in os.listdir(pdf_volume_path) if file not in processed_files]
processed_files

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

df.createOrReplaceTempView("temp_table") 
spark.sql("""
    INSERT INTO `genai-poc-catalog`.rag.docs_track
    SELECT * FROM temp_table
    WHERE NOT EXISTS (
        SELECT 1 FROM `genai-poc-catalog`.rag.docs_track
        WHERE temp_table.file_name = `genai-poc-catalog`.rag.docs_track.file_name
    )
""")


from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
import os

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="poc_scope", key="accessForDatabricksAPI")


# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="doc_vector_endpoint"
index_name="genai-poc-catalog.rag.doc_idx"

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host

    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )


    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()

# COMMAND ----------

import mlflow.pyfunc
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

class RAGPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load retriever and LLM inside the model context.
        """
        self.retriever = get_retriever()  # Load retriever function

        # Load Chat Model from Databricks Model Serving
        self.chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=200)

        # Define prompt template
        TEMPLATE = """You are an assistant for home appliance users. 
        You are answering how to, maintenance, and troubleshooting questions regarding the appliances you have data on. 
        If the question is not related, kindly decline to answer. 

        If you don't know the answer, just say so. Don't make up an answer. 
        Provide all answers only in English.

        Use the following pieces of context to answer the question at the end:
        {context}

        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

        # Define RetrievalQA chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    def predict(self, context, model_input):
        """
        Predict function for PyFunc model.
        Takes a Pandas DataFrame input and returns the model's response.
        """
        if isinstance(model_input, pd.DataFrame):
            query = model_input.iloc[0]["query"]  # Extract query from the first row
        else:
            query = model_input[0]  # Fallback for list input (during local testing)

        return self.chain.run({"query": query})


# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.models import infer_signature

# Set registry URI for Unity Catalog (ONLY IF USING UC)
mlflow.set_registry_uri("databricks-uc")

# Define model
model = RAGPyFuncModel()
model.load_context(None)  # Ensure retriever & LLM are loaded

# Define input example (convert to DataFrame)
input_example = pd.DataFrame({"query": ["What does a SUDS message mean?"]})

# Infer model signature
signature = infer_signature(input_example, model.predict(None, input_example))

# Define model name (Use underscores `_` for workspace registry)
registered_model_name = "genai-poc-catalog.rag.appliance_chatbot_pyfunc"

with mlflow.start_run(run_name="pyfunc_rag_run") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rag_pyfunc_model",
        python_model=model,
        registered_model_name=registered_model_name,  # Use correct name format
        pip_requirements=[
            "mlflow",
            "langchain",
            "langchain_community",
            "databricks-vectorsearch",
        ],
        input_example=input_example,  # Must be a DataFrame
        signature=signature,  # Required for Unity Catalog
    )

print(f"Model successfully logged and registered: {registered_model_name}")

import mlflow.pyfunc

model_name = "genai-poc-catalog.rag.appliance_chatbot_pyfunc"

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/7")

import pandas as pd

test_input = pd.DataFrame({"query": ["What does a SUDS message mean?"]})  

response = loaded_model.predict(test_input)  # Call the predict function
print(response)

dbutils.library.restartPython()

import os
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="poc_scope", key="accessForDatabricksAPI")
os.environ['API_ENDPOINT'] = "https://adb-2411896520795414.14.azuredatabricks.net/serving-endpoints/Chatbot_poc/invocations"


