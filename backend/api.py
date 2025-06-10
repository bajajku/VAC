from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from utils.helper import extract_sources_from_toolmessage
from core.app import get_app
from config.settings import get_settings
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.documents import Document
from datetime import datetime

# Enhanced imports for advanced RAG
from scripts.data_cleaning.data_cleaner import DataCleaner
from models.advanced_retriever import AdvancedRetriever
from models.tools.retriever_tool import configure_retriever, retrieve_information, get_retrieval_stats
from utils.retriever import global_retriever

# MongoDB and Feedback imports
from config.mongodb import mongodb_config
from models.feedback import (
    FeedbackCreate, FeedbackResponse, FeedbackUpdate, FeedbackStats,
    feedback_service
)

load_dotenv()

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    k: Optional[int] = 4

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    stats: Optional[dict] = None
    processing_info: Optional[dict] = None

class DocumentRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None
    use_enhanced_processing: Optional[bool] = True  # Enhanced processing by default

class StatusResponse(BaseModel):
    status: str
    message: str
    stats: Optional[dict] = None

class EnhancedQueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4
    retrieval_strategy: Optional[str] = "hybrid"  # similarity, mmr, hybrid, ensemble
    enable_reranking: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.7

class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback_type: str  # 'positive', 'negative', 'suggestion'
    feedback_text: Optional[str] = None
    rating: Optional[int] = None
    user_id: Optional[str] = None

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
advanced_retriever = None

@app_api.on_event("startup")
async def startup_event():
    """Initialize the enhanced RAG system and MongoDB on startup."""
    global rag_app, is_initialized, advanced_retriever
    
    try:
        # Initialize MongoDB connection
        print("🔌 Connecting to MongoDB...")
        mongodb_connected = await mongodb_config.connect()
        if not mongodb_connected:
            print("⚠️ MongoDB connection failed, feedback features will be disabled")
        
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        # Check if we should skip auto-processing on startup
        SKIP_AUTO_PROCESSING = os.getenv("SKIP_AUTO_PROCESSING", "false").lower() == "true"
        
        print(f"TOGETHER_API_KEY: {TOGETHER_API_KEY}")
        print(f"SKIP_AUTO_PROCESSING: {SKIP_AUTO_PROCESSING}")

        rag_app = get_app()
        
        config = {
            "app_type": "rag_agent",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_provider": "chatopenai",
            "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "persist_directory": "./chroma_db",
            "collection_name": "demo_collection",
            "api_key": TOGETHER_API_KEY,
            "chats_by_session_id": {}
        }
        rag_app.initialize(**config)

        # Initialize advanced retriever
        print("🚀 Initializing advanced retriever...")
        advanced_retriever = AdvancedRetriever(
            vector_store=rag_app.vector_db.vector_database,
            max_results=10,
            enable_reranking=True,
            similarity_threshold=0.7
        )
        print("✅ Advanced retriever initialized!")
        
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
        
        is_initialized = True
        print("🎉 Enhanced RAG API initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced RAG API: {e}")
        is_initialized = False

@app_api.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await mongodb_config.disconnect()

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
    """Query the RAG system with basic retrieval."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get the answer
        answer = await rag_app.aquery(request.question)
        
        # Get source documents using basic search
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

@app_api.post("/query-enhanced", response_model=QueryResponse)
async def query_rag_enhanced(request: EnhancedQueryRequest):
    """Query the RAG system with enhanced retrieval strategies."""
    if not is_initialized or not advanced_retriever:
        raise HTTPException(status_code=503, detail="Enhanced RAG system not initialized")
    
    try:
        # Configure advanced retriever
        config_result = configure_retriever.invoke({
            "max_results": request.k,
            "retrieval_strategy": request.retrieval_strategy,
            "enable_reranking": request.enable_reranking,
            "similarity_threshold": request.similarity_threshold
        })
        
        # Retrieve information with enhanced strategies
        retrieval_result = retrieve_information.invoke({
            "query": request.question
        })
        
        # Get statistics
        stats_result = get_retrieval_stats.invoke({})
        
        # Get the answer using enhanced retrieval context
        answer = await rag_app.aquery(request.question)
        
        # Format enhanced sources
        sources = []
        if 'retrieved_documents' in retrieval_result:
            for doc in retrieval_result['retrieved_documents']:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "retrieval_strategy": request.retrieval_strategy,
                        "similarity_score": doc.metadata.get('similarity_score', 'N/A'),
                        "content_type": doc.metadata.get('content_type', 'unknown')
                    }
                })
        
        return QueryResponse(
            answer=answer, 
            sources=sources,
            stats=stats_result.get('stats', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing enhanced query: {str(e)}")

@app_api.post("/stream_async")
async def stream_query(request: QueryRequest):
    """Stream the response to a query."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Generate session_id if not provided
    session_id = request.session_id
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    
    def event_stream():
        # Send session_id first
        yield f"data: [SESSION_ID]{session_id}[/SESSION_ID]\n\n"
        
        # Stream the response with session support
        for chunk in rag_app.stream_query(request.question, session_id):
            # Only stream the final AI messages (not tool calls or intermediate messages)
            
            if isinstance(chunk[0], AIMessage) and chunk[0].content and not chunk[0].tool_calls:
                yield f"data: {chunk[0].content}\n\n"
            
            if isinstance(chunk[0], ToolMessage):
                sources = extract_sources_from_toolmessage(chunk[0].content)
                for source in sources:
                    yield f"data: [SOURCE]{source}[/SOURCE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app_api.post("/documents")
async def add_documents(request: DocumentRequest):
    """Add documents to the vector database with optional enhanced processing."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        if request.use_enhanced_processing:
            # Use enhanced document processing
            print("🔄 Processing documents with enhanced cleaning and splitting...")
            
            # Create temporary JSON-like structure for DataCleaner
            temp_data = {}
            for i, (text, metadata) in enumerate(zip(request.texts, request.metadatas or [{}] * len(request.texts))):
                temp_data[f"doc_{i}"] = {
                    "text_content": text,
                    "title": metadata.get("title", f"Document {i}"),
                    "description": metadata.get("description", ""),
                    **metadata
                }
            
            # Clean and process with DataCleaner
            cleaner = DataCleaner(
                temp_data, 
                use_advanced_processing=True,
                chunk_size=800,
                chunk_overlap=100
            )
            enhanced_docs = cleaner.clean_data()
            
            # Add to vector database
            rag_app.vector_db.add_documents(enhanced_docs)
            
            return {
                "message": f"Added {len(enhanced_docs)} enhanced document chunks from {len(request.texts)} original documents",
                "processing_type": "enhanced",
                "original_count": len(request.texts),
                "final_count": len(enhanced_docs),
                "expansion_ratio": len(enhanced_docs) / len(request.texts) if request.texts else 0
            }
        else:
            # Use basic processing (backward compatibility)
            rag_app.add_documents_from_text(request.texts, request.metadatas)
            return {
                "message": f"Added {len(request.texts)} documents successfully",
                "processing_type": "basic"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@app_api.post("/load-json-enhanced")
async def load_json_data_enhanced(file_path: str, use_enhanced_processing: bool = True):
    """Load documents from a JSON file with enhanced processing."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        if use_enhanced_processing:
            print(f"📚 Loading {file_path} with enhanced processing...")
            
            # Load raw JSON data
            with open(file_path, 'r') as f:
                raw_data = f.read()
            
            # Process with DataCleaner
            cleaner = DataCleaner(
                raw_data,
                use_advanced_processing=True,
                chunk_size=800,
                chunk_overlap=100
            )
            enhanced_docs = cleaner.clean_data()
            
            # Add to vector database
            rag_app.vector_db.add_documents(enhanced_docs)
            
            # Get processing statistics
            stats = cleaner.doc_processor.get_processing_stats([], enhanced_docs) if cleaner.doc_processor else {}
            
            return {
                "message": f"Loaded and enhanced {len(enhanced_docs)} document chunks from {file_path}",
                "processing_type": "enhanced",
                "document_count": len(enhanced_docs),
                "processing_stats": stats
            }
        else:
            # Fallback to basic loading
            num_docs = rag_app.load_data_from_json(file_path)
            return {
                "message": f"Loaded {num_docs} documents from {file_path}",
                "processing_type": "basic"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading JSON data: {str(e)}")

@app_api.post("/load-json")
async def load_json_data(file_path: str):
    """Load documents from a JSON file with basic processing (backward compatibility)."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        num_docs = rag_app.load_data_from_json(file_path)
        return {"message": f"Loaded {num_docs} documents from {file_path}", "processing_type": "basic"}
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

@app_api.get("/retrieval-stats")
async def get_retrieval_statistics():
    """Get advanced retrieval statistics."""
    if not is_initialized or not advanced_retriever:
        raise HTTPException(status_code=503, detail="Enhanced RAG system not initialized")
    
    try:
        stats_result = get_retrieval_stats.invoke({})
        return stats_result.get('stats', {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting retrieval stats: {str(e)}")

@app_api.post("/configure-retriever")
async def configure_advanced_retriever(
    max_results: int = 5,
    retrieval_strategy: str = "hybrid",
    enable_reranking: bool = True,
    similarity_threshold: float = 0.7
):
    """Configure the advanced retriever settings."""
    if not is_initialized or not advanced_retriever:
        raise HTTPException(status_code=503, detail="Enhanced RAG system not initialized")
    
    try:
        result = configure_retriever.invoke({
            "max_results": max_results,
            "retrieval_strategy": retrieval_strategy,
            "enable_reranking": enable_reranking,
            "similarity_threshold": similarity_threshold
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring retriever: {str(e)}")

# =============================================================================
# MONGODB DEBUG ENDPOINTS
# =============================================================================

@app_api.get("/test-mongodb")
async def test_mongodb_connection():
    """Test MongoDB connection and provide detailed diagnostics."""
    diagnostics = {
        "connection_status": "unknown",
        "database_status": "unknown", 
        "collection_test": "unknown",
        "environment_vars": {},
        "error_details": None
    }
    
    try:
        # Check environment variables
        diagnostics["environment_vars"] = {
            "MONGODB_URL_SET": bool(os.getenv("MONGODB_URL")),
            "MONGODB_DATABASE_SET": bool(os.getenv("MONGODB_DATABASE")),
            "URL_PREFIX": os.getenv("MONGODB_URL", "")[:20] + "..." if os.getenv("MONGODB_URL") else None,
            "DATABASE_NAME": os.getenv("MONGODB_DATABASE", "not_set")
        }
        
        # Test connection
        print("🧪 Testing MongoDB connection...")
        mongodb_connected = await mongodb_config.connect()
        
        if mongodb_connected:
            diagnostics["connection_status"] = "✅ Connected successfully"
            
            # Test database access
            if mongodb_config.database is not None:
                diagnostics["database_status"] = f"✅ Database '{mongodb_config.database.name}' accessible"
                
                # Test collection operations
                try:
                    test_collection = mongodb_config.get_collection("connection_test")
                    
                    # Insert test document
                    test_doc = {
                        "test": True,
                        "timestamp": datetime.utcnow(),
                        "message": "Connection test successful"
                    }
                    result = await test_collection.insert_one(test_doc)
                    
                    # Count documents
                    count = await test_collection.count_documents({})
                    
                    # Clean up test document
                    await test_collection.delete_one({"_id": result.inserted_id})
                    
                    diagnostics["collection_test"] = f"✅ Insert/count/delete successful. Found {count} total test docs"
                    
                except Exception as e:
                    diagnostics["collection_test"] = f"❌ Collection operation failed: {str(e)}"
            else:
                diagnostics["database_status"] = "❌ Database object is None"
        else:
            diagnostics["connection_status"] = "❌ Connection failed"
            
    except Exception as e:
        diagnostics["error_details"] = str(e)
        diagnostics["connection_status"] = f"❌ Exception during connection test: {str(e)}"
    
    return diagnostics

@app_api.get("/feedback-service-status")
async def check_feedback_service_status():
    """Check if feedback service is ready to use."""
    try:
        # Check if MongoDB is connected
        if mongodb_config.database is None:
            return {
                "status": "❌ Not Ready",
                "issue": "MongoDB not connected",
                "solution": "Check MongoDB connection string and network access"
            }
        
        # Try to get feedback collection
        try:
            collection = mongodb_config.get_collection("feedback")
            count = await collection.count_documents({})
            
            return {
                "status": "✅ Ready",
                "feedback_collection": f"Accessible ({count} documents)",
                "database": mongodb_config.database.name
            }
        except Exception as e:
            return {
                "status": "❌ Not Ready", 
                "issue": f"Cannot access feedback collection: {str(e)}",
                "database_connected": mongodb_config.database is not None
            }
            
    except Exception as e:
        return {
            "status": "❌ Error",
            "error": str(e)
        }

# =============================================================================
# FEEDBACK ENDPOINTS
# =============================================================================

@app_api.post("/feedback", response_model=FeedbackResponse)
async def create_feedback(request: FeedbackRequest):
    """Create new feedback for a conversation."""
    try:
        feedback_data = FeedbackCreate(
            session_id=request.session_id,
            question=request.question,
            answer=request.answer,
            feedback_type=request.feedback_type,
            feedback_text=request.feedback_text,
            rating=request.rating,
            user_id=request.user_id
        )
        
        feedback = await feedback_service.create_feedback(feedback_data)
        return feedback
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating feedback: {str(e)}")

@app_api.get("/feedback/{feedback_id}", response_model=FeedbackResponse)
async def get_feedback(feedback_id: str):
    """Get feedback by ID."""
    try:
        feedback = await feedback_service.get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return feedback
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback: {str(e)}")

@app_api.get("/feedback/session/{session_id}", response_model=List[FeedbackResponse])
async def get_session_feedback(session_id: str):
    """Get all feedback for a session."""
    try:
        feedback_list = await feedback_service.get_feedback_by_session(session_id)
        return feedback_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session feedback: {str(e)}")

@app_api.put("/feedback/{feedback_id}", response_model=FeedbackResponse)
async def update_feedback(feedback_id: str, update_data: FeedbackUpdate):
    """Update feedback by ID."""
    try:
        feedback = await feedback_service.update_feedback(feedback_id, update_data)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return feedback
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating feedback: {str(e)}")

@app_api.delete("/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """Delete feedback by ID."""
    try:
        success = await feedback_service.delete_feedback(feedback_id)
        if not success:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return {"message": "Feedback deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting feedback: {str(e)}")

@app_api.get("/feedback-stats", response_model=FeedbackStats)
async def get_feedback_stats(limit: int = 10):
    """Get feedback statistics and recent feedback."""
    try:
        stats = await feedback_service.get_feedback_stats(limit)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback stats: {str(e)}")

# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=8000, reload=True) 