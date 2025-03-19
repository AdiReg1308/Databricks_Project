# RAG Chatbot on Databricks

## Overview

This repository contains the implementation of a RAG (Retrieval-Augmented Generation) chatbot using Databricks. It includes:

- A Gradio-based UI deployed via Databricks Apps.
- A Job scheduling notebook (`new_pdf_job_run.ipynb`) for automating document ingestion.
- Model serving using Databricks Vector Search and Unity Catalog.
- Integration with Mosaic AI Gateway for governance and security.

## Prerequisites

Before proceeding, ensure you have:

- A Databricks workspace with Unity Catalog enabled.
- A cluster with Unity Catalog access created and running.
- Databricks Apps enabled for deploying the Gradio-based UI.
- A serving endpoint configured for model inference.
- Service requests raised for Unity Catalog and Key Vault access (if needed).

## Setup Instructions

### 1. Deploy the Chatbot UI on Databricks Apps

1. Navigate to **Databricks → Apps**.
2. Click **Create App** and select **Gradio**.
3. Upload the `app/` folder.
4. Set the entry point as `main.py`.
5. Configure compute resources.
6. Deploy the app and obtain the public URL for access.

### 2. Set Up Document Ingestion Job

1. Open Databricks and navigate to **Workflows → Jobs**.
2. Click **Create Job**.
3. Upload the `new_pdf_job_run.ipynb` notebook.
4. Configure the job to trigger on file arrival in the document storage location.
5. Assign a cluster for execution and save the job.
6. Verify the job runs successfully and ingests documents into Vector Search.

### 3. Verify Model Serving Endpoint

1. Go to **Model Serving → Endpoints** in Databricks.
2. Ensure the serving endpoint for the chatbot model is active.
3. Test model inference using a sample query.

## Cost Considerations

- **Databricks Units (DBU)**: Costs are based on compute usage.
- **Vector Search**: Additional costs apply for indexing and querying.
- **Mosaic AI Gateway**: Logging and AI guardrails may incur extra usage costs.

## Next Steps

- Host an **Agentic Workflow** in the model endpoint for advanced workflows.
- Deploy **Spark-based Models** to optimize performance for larger datasets.
