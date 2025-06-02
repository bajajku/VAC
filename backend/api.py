from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from core.app import get_app
from config.settings import get_settings
import os
from dotenv import load_dotenv

load_dotenv()

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class DocumentRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    stats: Optional[dict] = None

# Initialize FastAPI app
app_api = FastAPI(
    title="RAG System API",
    description="API for Retrieval-Augmented Generation system using LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_app = None
is_initialized = False

@app_api.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_app, is_initialized
    
    try:
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        print(f"TOGETHER_API_KEY: {TOGETHER_API_KEY}")

        rag_app = get_app()
        # settings = get_settings()
        
        # # Validate configuration
        # issues = settings.validate_config()
        # if issues:
        #     print(f"⚠️ Configuration issues: {issues}")
        #     print("RAG system will start but may not work properly without proper configuration.")
        
        # Initialize with settings
        # rag_app.initialize(
        #     **settings.get_vector_db_kwargs(),
        #     app_type=settings.config.llm.app_type,
        #     llm_provider=settings.config.llm.provider,
        #     llm_model=settings.config.llm.model,
        #     **settings.get_llm_kwargs()
        # )
        config = {
        "app_type": "rag_agent",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_provider": "chatopenai",  # or "openrouter" for open source models
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "persist_directory": "./chroma_db",
        "collection_name": "demo_collection",
        "api_key": TOGETHER_API_KEY  # Uncomment and add your API key
        }
        rag_app.initialize(**config)

        
        # Check if there's existing JSON data to load
        json_files = list(Path("scripts/data_collection/crawl_results").glob("*.json"))
        if json_files:
            print(f"Found {len(json_files)} JSON files. Loading the most recent one...")
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            rag_app.load_data_from_json(str(latest_json))
        
        
        is_initialized = True
        print("✅ RAG API initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG API: {e}")
        is_initialized = False

@app_api.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint with system status."""
    if not is_initialized:
        return StatusResponse(
            status="error",
            message="RAG system not initialized"
        )
    
    stats = rag_app.get_stats()
    return StatusResponse(
        status="success",
        message="RAG System API is running",
        stats=stats
    )

@app_api.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if is_initialized else "unhealthy",
        "initialized": is_initialized
    }

@app_api.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get the answer
        answer = await rag_app.aquery(request.question)
        
        # Get source documents
        docs = rag_app.search_documents(request.question, k=request.k)
        sources = [
            {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# TODO: Implement streaming response
@app_api.post("/stream_async")
async def stream_query(request: QueryRequest):
    """Stream the response to a query."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    def event_stream():
        for chunk in rag_app.stream_query(request.question):
            yield chunk
            # Or: yield f"data: {chunk}\n\n"  # If using SSE

    return StreamingResponse(event_stream(), media_type="text/plain")


@app_api.post("/documents")
async def add_documents(request: DocumentRequest):
    """Add documents to the vector database."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        rag_app.add_documents_from_text(request.texts, request.metadatas)
        return {"message": f"Added {len(request.texts)} documents successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@app_api.post("/load-json")
async def load_json_data(file_path: str):
    """Load documents from a JSON file."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        num_docs = rag_app.load_data_from_json(file_path)
        return {"message": f"Loaded {num_docs} documents from {file_path}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading JSON data: {str(e)}")

@app_api.get("/search")
async def search_documents(query: str, k: int = 4):
    """Search for relevant documents without generating a response."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        docs = rag_app.search_documents(query, k=k)
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return {"query": query, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app_api.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_app.get_stats()

# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=8000) 