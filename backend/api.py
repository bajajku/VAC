from pathlib import Path
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from models.guardrails import Guardrails
from utils.helper import extract_sources_from_toolmessage
from core.app import get_app
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.documents import Document
from datetime import datetime
from fastapi import Request
from utils.rate_limiter import login_rate_limiter

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

# Chat Session imports
from models.chat_session import (
    ChatSessionCreate, ChatSessionResponse, ChatSessionUpdate, ChatMessage,
    chat_session_service
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from models.user import User, UserCreate, UserLogin, Token, TokenRefreshRequest, UserInDB, get_current_user, get_user_by_email
from utils.auth_service import (
    hash_password, verify_password, create_access_token, decode_access_token,
    create_refresh_token, decode_refresh_token, blacklist_token, is_password_complex
)
from utils.performance_monitor import performance_monitor, track_performance
from config.mongodb import mongodb_config
from utils.rate_limiter import cleanup_rate_limiter
import asyncio
import time

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

load_dotenv()

# Add these imports at the top
from config.oauth import get_google_oauth_config
from google.oauth2 import id_token
from google.auth.transport import requests
import httpx
from models.user import AuthProvider, OAuthUserInfo, OAuthState
import secrets
import json
from urllib.parse import urlencode

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
    
    # Detailed feedback categories (1-5 star ratings)
    retrieval_relevance: Optional[int] = None
    hallucination: Optional[int] = None
    noise_robustness: Optional[int] = None
    negative_rejection: Optional[int] = None
    privacy_breach: Optional[int] = None
    malicious_use: Optional[int] = None
    security_breach: Optional[int] = None
    out_of_domain: Optional[int] = None
    completeness: Optional[int] = None
    brand_damage: Optional[int] = None
    
    # Additional feedback fields
    vote: Optional[str] = None
    comment: Optional[str] = None
    expert_notes: Optional[str] = None

class NewSessionRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None

class NewSessionRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None

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

# =============================================================================
# AUTH ENDPOINTS
# =============================================================================
@app_api.post("/auth/register", response_model=Token)
async def register_user(user: UserCreate):
    """Register a new user."""
    try:
        users = mongodb_config.get_collection("users")
        
        # Check if email already exists
        existing = await users.find_one({"email": user.email})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
            
        # Validate password complexity
        is_complex, message = is_password_complex(user.password)
        if not is_complex:
            raise HTTPException(status_code=400, detail=message)
            
        # Hash password and prepare user data
        hashed_password = hash_password(user.password)
        now = datetime.utcnow()
        
        # Create user document
        await users.insert_one({
            "email": user.email, 
            "hashed_pw": hashed_password, 
            "name": user.username,
            "created_at": now, 
            "updated_at": now
        })
        
        # Generate tokens
        access_token = create_access_token({"sub": user.email})
        refresh_token = create_refresh_token({"sub": user.email})
        
        # Calculate expiry time for client info (seconds since epoch)
        expires_at = int(time.time()) + 60 * 60  # 1 hour from now
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_at=expires_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app_api.post("/auth/login", response_model=Token)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user and return JWT tokens."""
    # Check rate limiting
    client_ip = request.client.host
    if await login_rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts, please try again later"
        )
    
    try:
        users = mongodb_config.get_collection("users")
        user = await users.find_one({"email": form_data.username})
        
        # Validate credentials
        if not user or not verify_password(form_data.password, user["hashed_pw"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Generate tokens
        access_token = create_access_token({"sub": user["email"]})
        refresh_token = create_refresh_token({"sub": user["email"]})
        
        # Calculate expiry time for client info
        expires_at = int(time.time()) + 60 * 60  # 1 hour from now
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_at=expires_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )

@app_api.post("/auth/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefreshRequest):
    """Get a new access token using refresh token."""
    try:
        # Validate refresh token
        payload = decode_refresh_token(token_data.refresh_token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        # Verify user exists
        user = await get_user_by_email(mongodb_config, email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        # Generate new tokens
        new_access_token = create_access_token({"sub": email})
        new_refresh_token = create_refresh_token({"sub": email})
        
        # Calculate expiry time
        expires_at = int(time.time()) + 60 * 60  # 1 hour from now
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_at=expires_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh error: {str(e)}"
        )

@app_api.get("/auth/me", response_model=User)
async def get_current_user_endpoint(current_user: User = Depends(get_current_user)):
    """Get current authenticated user."""
    return current_user

@app_api.post("/auth/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout user and invalidate token."""
    try:
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Add token to blacklist
        blacklist_token(token)
        
        return {"message": "Logged out successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout error: {str(e)}"
        )

@app_api.post("/auth/password-reset-request")
async def request_password_reset(email: str):
    """Request password reset link."""
    # In a real application, you would:
    # 1. Generate a reset token
    # 2. Store token with expiry in database
    # 3. Send email with reset link
    
    # For this demo, we'll just simulate the process
    try:
        user = await get_user_by_email(email)
        if not user:
            # Don't reveal if email exists or not for security
            return {"message": "If the email exists, a reset link has been sent"}
            
        # In real application, send email here
        return {"message": "If the email exists, a reset link has been sent"}
    except Exception as e:
        # Still return same message to avoid revealing if email exists
        return {"message": "If the email exists, a reset link has been sent"}

@app_api.get("/auth/google/login")
async def google_login(redirect_url: str = "/"):
    """Start Google OAuth flow"""
    try:
        # Generate state token to prevent CSRF
        state_data = OAuthState(redirect_url=redirect_url, provider=AuthProvider.GOOGLE)
        state = secrets.token_urlsafe(32)
        
        # Store state in MongoDB (you might want to use Redis in production)
        states_collection = mongodb_config.get_collection("oauth_states")
        await states_collection.insert_one({
            "state": state,
            "data": json.loads(state_data.json()),
            "created_at": datetime.utcnow()
        })
        
        # Get Google OAuth config
        google_config = get_google_oauth_config()
        
        # Build authorization URL
        params = {
            "client_id": google_config.client_id,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": google_config.redirect_uri,
            "state": state,
            "access_type": "offline",  # To get refresh token
            "prompt": "consent"  # To always get refresh token
        }
        
        auth_url = f"{google_config.authorize_url}?{urlencode(params)}"
        return {"authorization_url": auth_url}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Google login: {str(e)}")

@app_api.get("/auth/google/callback")
async def google_callback(code: str, state: str):
    """Handle Google OAuth callback"""
    try:
        # Verify state token
        states_collection = mongodb_config.get_collection("oauth_states")
        stored_state = await states_collection.find_one({"state": state})
        
        if not stored_state:
            raise HTTPException(status_code=400, detail="Invalid state parameter")
            
        # Clean up used state
        await states_collection.delete_one({"state": state})
        
        # Get Google OAuth config
        google_config = get_google_oauth_config()
        
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                google_config.token_url,
                data={
                    "client_id": google_config.client_id,
                    "client_secret": google_config.client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": google_config.redirect_uri
                }
            )
            
            if token_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get token from Google")
                
            token_data = token_response.json()
            
            # Verify ID token
            id_info = id_token.verify_oauth2_token(
                token_data["id_token"],
                requests.Request(),
                google_config.client_id
            )
            
            # Get user info
            userinfo_response = await client.get(
                google_config.userinfo_url,
                headers={"Authorization": f"Bearer {token_data['access_token']}"}
            )
            
            if userinfo_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get user info from Google")
                
            userinfo = userinfo_response.json()
            
            # Create OAuth user info
            oauth_info = OAuthUserInfo(
                provider=AuthProvider.GOOGLE,
                provider_user_id=id_info["sub"],
                email=userinfo["email"],
                name=userinfo.get("name"),
                picture=userinfo.get("picture"),
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                expires_at=int(time.time()) + token_data["expires_in"]
            )
            
            # Get or create user
            users = mongodb_config.get_collection("users")
            user = await users.find_one({"email": oauth_info.email})
            
            if user:
                # Update existing user
                await users.update_one(
                    {"_id": user["_id"]},
                    {
                        "$set": {
                            "auth_provider": AuthProvider.GOOGLE,
                            "oauth_info": json.loads(oauth_info.json()),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            else:
                # Create new user
                user = {
                    "email": oauth_info.email,
                    "username": oauth_info.name or oauth_info.email.split("@")[0],
                    "auth_provider": AuthProvider.GOOGLE,
                    "oauth_info": json.loads(oauth_info.json()),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                await users.insert_one(user)
            
            # Create access token
            access_token = create_access_token({"sub": oauth_info.email})
            refresh_token = create_refresh_token({"sub": oauth_info.email})
            
            # Get redirect URL from state
            redirect_url = stored_state["data"]["redirect_url"]
            
            # Add tokens to URL
            if "?" in redirect_url:
                redirect_url += "&"
            else:
                redirect_url += "?"
            redirect_url += urlencode({
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            })
            
            return {"redirect_url": redirect_url}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Google callback: {str(e)}")

@app_api.post("/auth/google/refresh")
async def refresh_google_token(user: User = Depends(get_current_user)):
    """Refresh Google OAuth tokens"""
    try:
        if user.auth_provider != AuthProvider.GOOGLE or not user.oauth_info or not user.oauth_info.refresh_token:
            raise HTTPException(status_code=400, detail="No Google refresh token available")
            
        google_config = get_google_oauth_config()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                google_config.token_url,
                data={
                    "client_id": google_config.client_id,
                    "client_secret": google_config.client_secret,
                    "refresh_token": user.oauth_info.refresh_token,
                    "grant_type": "refresh_token"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to refresh Google token")
                
            token_data = response.json()
            
            # Update user's OAuth info
            users = mongodb_config.get_collection("users")
            oauth_info = user.oauth_info.copy()
            oauth_info.access_token = token_data["access_token"]
            oauth_info.expires_at = int(time.time()) + token_data["expires_in"]
            
            await users.update_one(
                {"email": user.email},
                {
                    "$set": {
                        "oauth_info": json.loads(oauth_info.json()),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return {"access_token": token_data["access_token"]}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh Google token: {str(e)}")

@app_api.on_event("startup")
async def startup_event():
    """Initialize the enhanced RAG system and MongoDB on startup."""
    global rag_app, is_initialized, advanced_retriever
    
    try:
        # Start background tasks
        asyncio.create_task(cleanup_rate_limiter())
        
        # Start session cache cleanup task
        async def periodic_cleanup():
            while True:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                await cleanup_session_cache()
        
        asyncio.create_task(periodic_cleanup())
        
        # Initialize MongoDB connection
        print("🔌 Connecting to MongoDB...")
        mongodb_connected = await mongodb_config.connect()
        if not mongodb_connected:
            print("⚠️ MongoDB connection failed, feedback features will be disabled")
            
        # Initialize chat session indexes
        print("📝 Initializing chat session service...")
        if await chat_session_service.initialize():
            print("✅ Chat session service initialized")
        else:
            print("⚠️ Failed to initialize chat session service")
            
        # Create OAuth states index with TTL
        print("🔑 Creating OAuth indexes...")
        states_collection = mongodb_config.get_collection("oauth_states")
        await states_collection.create_index("created_at", expireAfterSeconds=3600)  # Expire after 1 hour
        print("✅ OAuth indexes created")
        
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        # Check if we should skip auto-processing on startup
        SKIP_AUTO_PROCESSING = os.getenv("SKIP_AUTO_PROCESSING", "false").lower() == "true"
        
        print(f"TOGETHER_API_KEY: {TOGETHER_API_KEY}")
        print(f"SKIP_AUTO_PROCESSING: {SKIP_AUTO_PROCESSING}")

        rag_app = get_app()
        
        config = {
            "app_type": "rag_agent",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_provider": "ollama",
            "llm_model": "llama3.1:70b",
            "persist_directory": "./chroma_db",
            "collection_name": "demo_collection",
            "api_key": TOGETHER_API_KEY,
            "chats_by_session_id": {},
            "input_guardrails": Guardrails().with_policy("maximum_protection"),
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
                print(f"📚 Found {len(cleaned_files)} preprocessed cleaned files. Loading all files...")
                total_documents = 0
                
                for cleaned_file in cleaned_files:
                    print(f"📁 Loading preprocessed data from: {cleaned_file}")
                    try:
                        # Load preprocessed cleaned data directly
                        import json
                        with open(cleaned_file, 'r') as f:
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
                        total_documents += len(documents)
                        print(f"✅ Loaded {len(documents)} preprocessed document chunks from {cleaned_file}")
                    except Exception as e:
                        print(f"⚠️ Failed to load preprocessed data from {cleaned_file}: {e}")
                        print(f"🔍 Debug info: File exists={cleaned_file.exists()}, Size={cleaned_file.stat().st_size if cleaned_file.exists() else 'N/A'}")
                
                print(f"📊 Total documents loaded from all files: {total_documents}")
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
        answer = await rag_app.aquery(request.question, session_id=request.session_id)
        
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

# =============================================================================
# CHAT SESSION ENDPOINTS
# =============================================================================

@app_api.post("/sessions/new", response_model=ChatSessionResponse)
async def create_new_session(
    request: NewSessionRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new chat session tied to authenticated user."""
    try:
        session_data = ChatSessionCreate(
            session_id=str(uuid.uuid4()),
            user_id=current_user.email,  # Always use authenticated user
            title=request.title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        session = await chat_session_service.create_session(session_data)
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app_api.get("/sessions", response_model=List[ChatSessionResponse])
async def list_sessions(
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """List chat sessions for the authenticated user."""
    try:
        sessions = await chat_session_service.list_sessions(
            user_id=current_user.email,
            limit=limit
        )
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app_api.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific chat session with security check."""
    try:
        session = await chat_session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Security: Verify ownership
        if session.user_id != current_user.email:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
            
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app_api.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all messages for a specific session with security check."""
    try:
        session = await chat_session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Security: Verify ownership
        if session.user_id != current_user.email:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
            
        messages = await chat_session_service.get_session_messages(session_id)
        return messages
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session messages: {str(e)}")

@app_api.put("/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session(
    session_id: str,
    update_data: ChatSessionUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a chat session with security check."""
    try:
        session = await chat_session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Security: Verify ownership
        if session.user_id != current_user.email:
            raise HTTPException(status_code=403, detail="Not authorized to modify this session")
            
        updated_session = await chat_session_service.update_session(session_id, update_data)
        if not updated_session:
            raise HTTPException(status_code=500, detail="Failed to update session")
            
        return updated_session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

@app_api.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a chat session with security check."""
    try:
        session = await chat_session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Security: Verify ownership
        if session.user_id != current_user.email:
            raise HTTPException(status_code=403, detail="Not authorized to delete this session")
            
        success = await chat_session_service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete session")
            
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app_api.get("/sessions/user/stats")
async def get_user_sessions_stats(current_user: User = Depends(get_current_user)):
    """Get statistics about user's sessions."""
    try:
        user_sessions = await chat_session_service.list_sessions(user_id=current_user.email)
        total_messages = 0
        
        for session in user_sessions:
            messages = await chat_session_service.get_session_messages(session.session_id)
            total_messages += len(messages)
            
        return {
            "total_sessions": len(user_sessions),
            "total_messages": total_messages,
            "oldest_session": min(user_sessions, key=lambda s: s.created_at).created_at if user_sessions else None,
            "newest_session": max(user_sessions, key=lambda s: s.created_at).created_at if user_sessions else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session stats: {str(e)}")

async def get_or_create_default_session(user_email: str) -> str:
    """Get user's default session or create one if none exists."""
    try:
        sessions = await chat_session_service.list_sessions(user_id=user_email, limit=1)
        if sessions:
            return sessions[0].session_id
        
        # Create default session
        session_data = ChatSessionCreate(
            session_id=str(uuid.uuid4()),
            user_id=user_email,
            title="Default Chat"
        )
        session = await chat_session_service.create_session(session_data)
        return session.session_id
    except Exception as e:
        print(f"Error in get_or_create_default_session: {e}")
        raise

# Session cache to avoid repeated DB queries
session_cache = {}
cache_ttl = 300  # 5 minutes TTL

async def cleanup_session_cache():
    """Cleanup expired session cache entries."""
    current_time = time.time()
    expired_keys = []
    
    for key, (_, timestamp) in session_cache.items():
        if current_time - timestamp > cache_ttl:
            expired_keys.append(key)
    
    for key in expired_keys:
        del session_cache[key]
    
    print(f"Cleaned up {len(expired_keys)} expired session cache entries")

@track_performance("session_validation_time")
async def validate_session_cached(session_id: str, user_email: str) -> bool:
    """Validate session with caching to avoid repeated DB queries."""
    cache_key = f"{session_id}:{user_email}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in session_cache:
        cached_data, timestamp = session_cache[cache_key]
        if current_time - timestamp < cache_ttl:
            return cached_data
        else:
            # Remove expired cache entry
            del session_cache[cache_key]
    
    # Query database
    try:
        session = await chat_session_service.get_session(session_id)
        is_valid = session and session.user_id == user_email
        
        # Cache the result
        session_cache[cache_key] = (is_valid, current_time)
        return is_valid
    except Exception as e:
        print(f"Session validation error: {e}")
        return False

@track_performance("db_operation_time")
async def store_user_message_background(session_id: str, question: str):
    """Store user message in background."""
    try:
        user_message = ChatMessage(
            content=question,
            sender="user",
            timestamp=datetime.utcnow()
        )
        await chat_session_service.add_message(session_id, user_message)
    except Exception as e:
        print(f"Warning: Could not store user message: {e}")

@track_performance("db_operation_time")
async def store_ai_message_background(session_id: str, response: str, sources: List[str]):
    """Store AI message in background."""
    try:
        if response.strip():
            ai_message = ChatMessage(
                content=response,
                sender="assistant",
                timestamp=datetime.utcnow(),
                sources=sources
            )
            await chat_session_service.add_message(session_id, ai_message)
    except Exception as e:
        print(f"Warning: Could not store AI message: {e}")

# Update the stream_query endpoint to use authentication
@app_api.post("/stream_async")
async def stream_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Stream the response to a query with user authentication and optimized performance."""
    init_start_time = time.time()
    
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Fast session handling
    session_id = request.session_id
    if not session_id:
        session_id = await get_or_create_default_session(current_user.email)
    else:
        # Use cached session validation
        is_valid = await validate_session_cached(session_id, current_user.email)
        if not is_valid:
            session_id = await get_or_create_default_session(current_user.email)
    
    # Record stream initialization time
    init_time = time.time() - init_start_time
    performance_monitor.record_metric("stream_init_time", init_time)
    
    async def event_stream():
        stream_start_time = time.time()
        first_chunk_sent = False
        
        # Send session_id first
        yield f"data: [SESSION_ID]{session_id}[/SESSION_ID]\n\n"
        
        # Store user message
        try:
            user_message = ChatMessage(
                content=request.question,
                sender="user",
                timestamp=datetime.utcnow()
            )
            await chat_session_service.add_message(session_id, user_message)
        except Exception as e:
            print(f"Warning: Could not store user message: {e}")
        
<<<<<<< HEAD
        # Collect the full AI response and sources
        full_response = ""
        collected_sources = []
=======
        # Collect the full AI response
        full_response = ""
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
        
        # Start retrieval timing
        performance_monitor.start_timer("retrieval_time")
        
        # Stream the response with optimized configuration
        for chunk in rag_app.stream_query(request.question, session_id):
            if isinstance(chunk[0], AIMessage) and chunk[0].content and not chunk[0].tool_calls:
                content = chunk[0].content
                full_response += content
                
                # Track first chunk time
                if not first_chunk_sent:
                    first_chunk_time = time.time() - stream_start_time
                    performance_monitor.record_metric("first_chunk_time", first_chunk_time)
                    first_chunk_sent = True
                
                yield f"data: {content}\n\n"
            
            if isinstance(chunk[0], ToolMessage):
                # End retrieval timing when we get tool results
                performance_monitor.end_timer("retrieval_time")
                
                sources = extract_sources_from_toolmessage(chunk[0].content)
                for source in sources:
                    if source and source not in collected_sources:
                        collected_sources.append(source)
                    print(source)
                    yield f"data: [SOURCE]{source}[/SOURCE]\n\n"
        
<<<<<<< HEAD
        # Store AI response with sources
=======
        # Store AI response
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
        try:
            if full_response.strip():
                ai_message = ChatMessage(
                    content=full_response,
                    sender="assistant",
<<<<<<< HEAD
                    timestamp=datetime.utcnow(),
                    sources=collected_sources
=======
                    timestamp=datetime.utcnow()
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
                )
                await chat_session_service.add_message(session_id, ai_message)
        except Exception as e:
            print(f"Warning: Could not store AI message: {e}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Update the stream_query endpoint to use authentication
@app_api.post("/stream_async_test")
async def stream_query_test(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
):
    """Stream the response to a query with user authentication and optimized performance."""
    init_start_time = time.time()
    
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Record stream initialization time
    init_time = time.time() - init_start_time
    performance_monitor.record_metric("stream_init_time", init_time)
    
    async def event_stream():
        stream_start_time = time.time()
        first_chunk_sent = False
                
        # Store user message in background (non-blocking)
        background_tasks.add_task(store_user_message_background, "test-session-id", request.question)
        
        # Collect the full AI response and sources
        full_response = ""
        collected_sources = []
        
        # Start retrieval timing
        performance_monitor.start_timer("retrieval_time")
        
        # Stream the response with optimized configuration
        for chunk in rag_app.stream_query(request.question, "test-session-id"):
            if isinstance(chunk[0], AIMessage) and chunk[0].content and not chunk[0].tool_calls:
                content = chunk[0].content
                full_response += content
                
                # Track first chunk time
                if not first_chunk_sent:
                    first_chunk_time = time.time() - stream_start_time
                    performance_monitor.record_metric("first_chunk_time", first_chunk_time)
                    first_chunk_sent = True
                
                print(content)
                yield f"data: {content}\n\n"
            
            if isinstance(chunk[0], ToolMessage):
                # End retrieval timing when we get tool results
                performance_monitor.end_timer("retrieval_time")
                
                sources = extract_sources_from_toolmessage(chunk[0].content)
                for source in sources:
                    if source and source not in collected_sources:
                        collected_sources.append(source)
                    yield f"data: [SOURCE]{source}[/SOURCE]\n\n"
        
        # Record total streaming time
        total_time = time.time() - stream_start_time
        performance_monitor.record_metric("total_stream_time", total_time)
        
        # Store AI response in background (non-blocking)
        background_tasks.add_task(store_ai_message_background, "test-session-id", full_response, collected_sources)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app_api.post("/stream_async_optimized")
async def stream_query_optimized(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """Highly optimized streaming endpoint with minimal overhead."""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    async def event_stream():
        # Minimal session handling - use cached session or fallback
        session_id = request.session_id or f"temp-{current_user.email}-{int(time.time())}"
        
        # Send session_id immediately
        yield f"data: [SESSION_ID]{session_id}[/SESSION_ID]\n\n"
        
        # Direct streaming with minimal processing
        try:
            for chunk in rag_app.stream_query(request.question, session_id):
                if isinstance(chunk[0], AIMessage) and chunk[0].content and not chunk[0].tool_calls:
                    # Direct yield without extra processing
                    yield f"data: {chunk[0].content}\n\n"
                elif isinstance(chunk[0], ToolMessage):
                    # Quick source extraction
                    sources = extract_sources_from_toolmessage(chunk[0].content)
                    for source in sources:
                        if source:
                            yield f"data: [SOURCE]{source}[/SOURCE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
            user_id=request.user_id,
            
            # Pass all detailed feedback fields from the request
            retrieval_relevance=request.retrieval_relevance,
            hallucination=request.hallucination,
            noise_robustness=request.noise_robustness,
            negative_rejection=request.negative_rejection,
            privacy_breach=request.privacy_breach,
            malicious_use=request.malicious_use,
            security_breach=request.security_breach,
            out_of_domain=request.out_of_domain,
            completeness=request.completeness,
            brand_damage=request.brand_damage,
            
            # Pass additional feedback fields
            vote=request.vote,
            comment=request.comment,
            expert_notes=request.expert_notes
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

@app_api.get("/performance-stats")
async def get_performance_stats():
    """Get performance statistics for streaming optimizations."""
    try:
        stats = performance_monitor.get_all_stats()
        summary = performance_monitor.get_performance_summary()
        
        return {
            "detailed_stats": stats,
            "summary": summary,
            "cache_stats": {
                "session_cache_size": len(session_cache),
                "cache_ttl_seconds": cache_ttl
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving performance stats: {str(e)}")

# =============================================================================
# CHAT SESSION ENDPOINTS
# =============================================================================

# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=8000, reload=True) 