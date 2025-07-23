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

**Success Response (200 OK):**

```json
{
    "answer": "Post-Traumatic Stress Disorder (PTSD) can manifest through symptoms like flashbacks, nightmares, and severe anxiety.",
    "sources": ["mental_health_guide_ptsd.pdf"],
    "session_id": "user123_session_abc"
}
```
-   `answer` (string): The agent's response to the question.
-   `sources` (list of strings): A list of source identifiers for the information used to generate the answer.
-   `session_id` (string): The session identifier for the conversation.


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

## 4. 🛡️ Managing and Testing Guardrails

This minimal API comes with input guardrails pre-configured to ensure safety and appropriate responses. Here’s how you can manage and test them.

### How Guardrails Are Loaded

The API initializes guardrails using a policy defined in `minimal_api.py`:

```python
# in minimal_api.py
"input_guardrails": Guardrails().with_policy("maximum_protection")
```

The `"maximum_protection"` policy activates a set of guardrail validators defined in `backend/config/guardrails_config.py`.

### How to Update or Add Validators

All guardrail configurations are centralized in `backend/config/guardrails_config.py`.

**1. To update an existing validator:**

Simply change its parameters. For example, to make the `ToxicLanguage` guard more sensitive, lower its `threshold`:

```python
# in backend/config/guardrails_config.py

"profanity_hate_harassment": [
    (ToxicLanguage, {
        "threshold": 0.5,  # Changed from 0.7 to 0.5 for higher sensitivity
        "on_fail": "exception"
    }),
    # ...
],
```

**2. To add a new validator to a policy:**

First, add the validator to a category in `GUARDRAIL_CONFIG`. For example, to add a `RegexMatch` to block prompts that ask the agent to ignore rules:

```python
# in backend/config/guardrails_config.py
from guardrails.hub import RegexMatch # Make sure to import it

# ...

"jailbreak": [
    (DetectJailbreak, { ... }),
    (RegexMatch, { # New validator added
        "regex": "(ignore|disregard) all previous instructions",
        "match_type": "search",
        "on_fail": "exception"
    })
],
```

Then, ensure the category (`jailbreak` in this case) is part of the policy you are using (e.g., `maximum_protection`) in the `GUARDRAIL_POLICIES` dictionary within the same file.

### How to Change the Active Policy

You can easily switch to a different pre-defined policy (e.g., `"military_mental_health"` or `"performance_optimized"`) by editing this line in `minimal_api.py`:

```python
# in minimal_api.py
"input_guardrails": Guardrails().with_policy("military_mental_health") # Changed policy
```

### Testing Guardrails via the API

You can test if the guardrails are working by sending prompts to the `/query` endpoint that should trigger a violation. The agent is configured to return a safe, pre-defined "fallback" message instead of a harmful response or an error.

**Example: Testing for Toxicity**

This prompt should be blocked by the `ToxicLanguage` validator.

```bash
curl -X POST "http://localhost:8001/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "You are a f***ing idiot",
  "session_id": "guardrail-test-1"
}'
```

**Expected Fallback Response:**

Instead of an answer from the LLM, you will receive a generic, safe response from the fallback system, confirming the guardrail worked:

```json
{
  "answer": "I cannot respond to this topic. Please ask a question related to military mental health support.",
  "sources": [],
  "session_id": "guardrail-test-1"
}
```

## 5. Automatic Data Loading on Startup

The enhanced `minimal_api.py` now includes an intelligent data loading mechanism when it starts:

1.  **Checks for Preprocessed Data**: It first looks in `backend/scripts/data_cleaning/cleaned_data/` for pre-cleaned and chunked JSON files. If found, it loads the latest one directly into the vector database. This is the fastest and most efficient method.
2.  **Fallback to Raw Data**: If no preprocessed data is found, it falls back to looking for raw data in `backend/scripts/data_collection/crawl_results/`. It loads the latest raw JSON file with basic processing.
3.  **Manual Loading**: If no data is found, the API will start with an empty knowledge base, and you can use the `/add_documents` endpoint to add data manually.

For best performance, run the offline preprocessing script (`python preprocess_data.py --auto`) before starting the API.

## 6. Local Vector Database

This minimal API creates its own local vector database in the `backend/minimal_chroma_db` directory. This ensures it does not interfere with the main application's database. You can safely delete this directory to clear the knowledge base and start fresh. 