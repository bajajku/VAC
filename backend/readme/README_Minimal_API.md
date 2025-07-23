# Minimal RAG Agent API - README

This document provides instructions on how to set up and run the minimalist API for the RAG (Retrieval-Augmented Generation) agent. This API exposes the core functionalities of the agent without the complexities of the main application, making it ideal for quick testing, integration, and focused use cases.

## Features

- **Query Endpoint**: Ask a question and get a response from the RAG agent.
- **Document Ingestion Endpoint**: Add new documents to the agent's knowledge base on the fly.
- **Stateful Conversations**: Supports session IDs to maintain conversation context.
- **Easy Setup**: Runs as a standalone FastAPI application with minimal configuration.
- **Health Check**: An endpoint to verify the status of the API and RAG agent.

## Prerequisites

- Python 3.10+
- All dependencies listed in `backend/requirements.txt` installed.
- An environment file (`.env`) in the project root directory.

## 1. Configuration

1.  **Copy the Environment File**: If you haven't already, copy the example environment file to create your own configuration:
    ```bash
    cp env.example .env
    ```

2.  **Set Your API Key**: Open the `.env` file and add your `TOGETHER_API_KEY`. The minimal API is configured to use Together AI by default.
    ```
    TOGETHER_API_KEY="your-xxxxxxxx-api-key"
    ```

## 2. Running the API

You can run the API directly using `uvicorn` or by executing the Python script.

### Using Python

From the `backend` directory, run:

```bash
python minimal_api.py
```

### Using Uvicorn

From the `backend` directory, run:

```bash
uvicorn minimal_api:app --reload --port 8001
```

The API will start and be accessible at `http://localhost:8001`.

## 3. API Endpoints

The API provides the following endpoints. You can also view interactive documentation by navigating to `http://localhost:8001/docs` in your browser.

### POST `/query`

Send a question to the RAG agent and receive an answer.

**Request Body:**

```json
{
  "question": "What are the symptoms of PTSD?",
  "session_id": "user123_session_abc"
}
```

-   `question` (string, required): The question you want to ask the agent.
-   `session_id` (string, optional): An identifier for the conversation session. This allows the agent to remember context from previous turns in the same session. Defaults to `"default_session"`.

**Example `curl` Request:**

```bash
curl -X POST "http://localhost:8001/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "What is trauma-informed care?",
  "session_id": "my-test-session"
}'
```

### POST `/add_documents`

Add new documents to the RAG agent's knowledge base.

**Request Body:**

```json
{
  "texts": [
    "This is the content of the first document.",
    "This is the content of the second document."
  ],
  "metadatas": [
    { "source": "doc_source_1" },
    { "source": "doc_source_2" }
  ]
}
```

-   `texts` (list of strings, required): The content of the documents you want to add.
-   `metadatas` (list of dicts, optional): Corresponding metadata for each document. The length of this list must match the length of `texts`.

**Example `curl` Request:**

```bash
curl -X POST "http://localhost:8001/add_documents" \
-H "Content-Type: application/json" \
-d '{
  "texts": ["Cognitive Behavioral Therapy (CBT) is a common treatment for anxiety."],
  "metadatas": [{ "source": "psychology_today", "topic": "cbt" }]
}'
```

### GET `/health`

Check the operational status of the API.

**Example `curl` Request:**

```bash
curl -X GET "http://localhost:8001/health"
```

Returns a simple JSON object indicating the status.

## 4. Local Vector Database

This minimal API creates its own local vector database in the `backend/minimal_chroma_db` directory. This ensures it does not interfere with the main application's database. You can safely delete this directory to clear the knowledge base and start fresh. 