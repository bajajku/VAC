import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from dotenv import load_dotenv

from models.guardrails import Guardrails

# Add backend directory to path
# This ensures that we can import modules from the project's backend.
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

try:
    from core.app import RAGApplication
    from langchain_core.runnables import RunnableConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this from the 'backend' directory or have the path set up correctly.")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv(dotenv_path=backend_dir.parent / '.env')

# --- Configuration ---
API_KEY = os.environ.get('TOGETHER_API_KEY')
LLM_PROVIDER = "chatopenai"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./minimal_chroma_db"
COLLECTION_NAME = "minimal_rag_collection"
SKIP_AUTO_PROCESSING = False

# --- FastAPI App ---
app = FastAPI(
    title="Minimal RAG Agent API",
    description="A minimalist API for interacting with the core functionality of the RAG Agent.",
    version="1.0.0"
)

# Global RAG application instance
rag_app = RAGApplication()

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

class DocumentRequest(BaseModel):
    texts: list[str]
    metadatas: list[dict] = []

class DocumentResponse(BaseModel):
    message: str
    documents_added: int

# --- API Events ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG Application on API startup.
    """
    print("🚀 API starting up...")
    if not API_KEY:
        print("FATAL: TOGETHER_API_KEY not found in .env file. The API will not work.")
        # We don't raise an exception here to allow the server to start,
        # but endpoints will fail gracefully.
        return

    config = {
        "app_type": "rag_agent",
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "persist_directory": PERSIST_DIRECTORY,
        "collection_name": COLLECTION_NAME,
        "api_key": API_KEY,
        "chats_by_session_id": {},  # Required for stateful conversations
        "input_guardrails": Guardrails().with_policy("maximum_protection")
    }
    
    try:
        rag_app.initialize(**config)
        print(f"✅ RAG Application initialized successfully with collection: '{COLLECTION_NAME}'")

        # Only auto-load data if not skipping auto-processing
        if not SKIP_AUTO_PROCESSING:
            print("🔄 Auto-processing enabled. Checking for data files...")
            
            # Check for preprocessed cleaned data first
            cleaned_data_dir = Path("scripts/data_cleaning/cleaned_data")
            # Filter out _info.json files - only get actual data files
            cleaned_files = [f for f in cleaned_data_dir.glob("*.json") if not f.name.endswith("_info.json")] if cleaned_data_dir.exists() else []
            
            if cleaned_files:
                print(f"📚 Found {len(cleaned_files)} preprocessed cleaned files. Loading...")
                latest_cleaned = max(cleaned_files, key=os.path.getctime)
                print(f"📁 Loading preprocessed data from: {latest_cleaned}")
                try:
                    # Load preprocessed cleaned data directly
                    import json
                    with open(latest_cleaned, 'r') as f:
                        cleaned_data = json.load(f)
                    
                    # Validate the data structure
                    if not isinstance(cleaned_data, list):
                        raise ValueError(f"Expected list of documents, got {type(cleaned_data)}")
                    
                    if not cleaned_data:
                        raise ValueError("No documents found in preprocessed file")
                    
                    # Verify first item has expected structure
                    if not isinstance(cleaned_data[0], dict) or 'page_content' not in cleaned_data[0]:
                        raise ValueError("Invalid document structure in preprocessed file")
                    
                    # Convert to documents and add to vector DB
                    documents = []
                    for item in cleaned_data:
                        doc = Document(
                            page_content=item['page_content'],
                            metadata=item['metadata']
                        )
                        documents.append(doc)
                    
                    rag_app.vector_db.add_documents(documents)
                    print(f"✅ Loaded {len(documents)} preprocessed document chunks")
                except Exception as e:
                    print(f"⚠️ Failed to load preprocessed data from {latest_cleaned}: {e}")
                    print(f"🔍 Debug info: File exists={latest_cleaned.exists()}, Size={latest_cleaned.stat().st_size if latest_cleaned.exists() else 'N/A'}")
            else:
                # Fallback to raw data processing (only if no cleaned data available)
                print("📁 No preprocessed data found. Checking for raw data...")
                json_files = list(Path("scripts/data_collection/crawl_results").glob("*.json"))
                if json_files:
                    latest_file = max(json_files, key=os.path.getctime)
                    print(f"📚 Loading raw data: {latest_file}")
                    try:
                        # Basic loading only (no expensive cleaning)
                        num_docs = rag_app.load_data_from_json(str(latest_file))
                        print(f"✅ Loaded {num_docs} documents with basic processing")
                        print("💡 Tip: Use 'python preprocess_data.py' to clean data offline for better performance")
                    except Exception as e:
                        print(f"⚠️ Failed to load raw data: {e}")
        else:
            print("⏩ Auto-processing skipped. Use API endpoints to load data manually.")
        
        print("🎉 Enhanced RAG API initialized successfully!")
    except Exception as e:
        print(f"❌ Error during RAG application initialization: {e}")
        # The app will continue to run, but endpoints will be aware of the uninitialized state.

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Receives a question and returns an answer from the RAG agent.
    """
    if not rag_app.is_initialized:
        raise HTTPException(status_code=503, detail="RAG application is not initialized. Check API logs for errors.")

    try:
        # Invoke the RAG agent
        response, sources = rag_app.query(request.question, session_id=request.session_id)
        print(f"Response: {response}")
        print(f"Sources: {sources}")
        
        return QueryResponse(answer=response, sources=sources, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")

@app.post("/add_documents", response_model=DocumentResponse)
async def add_documents(request: DocumentRequest):
    """
    Adds new documents to the vector database.
    """
    if not rag_app.is_initialized:
        raise HTTPException(status_code=503, detail="RAG application is not initialized. Check API logs for errors.")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="The 'texts' field cannot be empty.")

    try:
        count = rag_app.add_documents_from_text(request.texts, request.metadatas)
        return DocumentResponse(message="Documents added successfully.", documents_added=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while adding documents: {e}")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    if rag_app.is_initialized:
        return {"status": "ok", "message": "RAG application is initialized."}
    else:
        return {"status": "error", "message": "RAG application is not initialized."}

# --- Main entrypoint to run the API ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Minimal RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 