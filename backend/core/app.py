import os
from abc import ABC, abstractmethod
from typing import Optional
from models.vector_database import VectorDatabase
from models.rag_agent import RAGAgent, create_rag_agent
from utils.retriever import global_retriever
from scripts.data_collection.json_parser import JsonParser
from models.rag_chain import RAGChain
from models.llm import LLM
from langchain.schema import HumanMessage
from models.guardrails import Guardrails

class BaseRAGApplication(ABC):
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.vector_db: Optional[VectorDatabase] = None
        self.rag_application = None
        self.is_initialized = False
        self.input_guardrails: Optional[Guardrails] = None
        self.output_guardrails: Optional[Guardrails] = None
    
    @abstractmethod
    def initialize(self, **kwargs):
        pass
    def load_data_from_json(self, json_file_path: str) -> int:
        pass
    
    def add_documents_from_text(self, texts: list[str], metadatas: list[dict] = None):
        pass
    

class RAGApplication:
    """
    Main application class that manages the RAG system components.
    Handles initialization, configuration, and provides a unified interface.
    """
    

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.vector_db: Optional[VectorDatabase] = None
        self.rag_application = None
        self.is_initialized = False
        self.input_guardrails: Optional[Guardrails] = None
        self.output_guardrails: Optional[Guardrails] = None
    
    def initialize(
        self,
        app_type: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_collection",
        **kwargs
    ):
        """
        Initialize the RAG application with vector database and LLM.
        
        Args:
            embedding_model: HuggingFace embedding model name
            llm_provider: LLM provider ('openai', 'openrouter', 'huggingface_pipeline', etc.)
            llm_model: LLM model name
            persist_directory: Directory to persist vector database
            collection_name: Name for the vector database collection
            **kwargs: Additional configuration (api_key, temperature, prompt, etc.)
        """
        print("🚀 Initializing RAG Application...")
        
        # Initialize vector database
        self.vector_db = VectorDatabase()
        print(f"collection_name: {collection_name}")
        self.vector_db.create_vector_database(
            embedding_model=embedding_model,
            type="chroma",
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Initialize global retriever
        print("🔍 Setting up retriever...")
        global_retriever.initialize(self.vector_db)
        
        # Separate LLM kwargs from RAG application kwargs
        llm_kwargs = {k: v for k, v in kwargs.items() if k not in ['prompt', 'chats_by_session_id', 'input_guardrails', 'output_guardrails']}
        rag_kwargs = {k: v for k, v in kwargs.items() if k in ['prompt', 'chats_by_session_id', 'input_guardrails', 'output_guardrails']}
        
        # Initialize LLM
        print(f"🤖 Setting up LLM with {llm_provider}/{llm_model}...")
        self.llm = LLM(provider=llm_provider, model_name=llm_model, **llm_kwargs)

        # Initialize RAG application
        self.rag_application = RAGApplicationFactory.create_app(
            app_type=app_type,
            llm=self.llm,
            **rag_kwargs
        )
        
        self.is_initialized = True
        print("✅ RAG Application initialized successfully!")
    
    def load_data_from_json(self, json_file_path: str) -> int:
        """
        Load documents from a JSON file into the vector database.
        
        Args:
            json_file_path: Path to the JSON file containing crawled data
            
        Returns:
            int: Number of documents loaded
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        print(f"📚 Loading data from {json_file_path}...")
        num_docs = self.vector_db.load_documents_from_json(json_file_path)
        print(f"✅ Loaded {num_docs} documents into vector database")
        return num_docs
    
    def add_documents_from_text(self, texts: list[str], metadatas: list[dict] = None):
        """
        Add documents directly from text content.
        
        Args:
            texts: List of text content
            metadatas: Optional list of metadata dictionaries
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        from langchain_core.documents import Document
        
        if metadatas is None or len(metadatas) == 0:
            metadatas = [{}] * len(texts)
        
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        self.vector_db.add_documents(documents)
        print(f"✅ Added {len(documents)} documents to vector database")
    
    def query(self, question: str, session_id: str = None) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            session_id: Session ID
        Returns:
            str: Agent's response
        """
        config = {"configurable": {"session_id": session_id}}
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return self.rag_application.invoke(question, config=config)
    
    async def aquery(self, question: str, session_id: str = None) -> str:
        """Async version of query."""
        config = {"configurable": {"session_id": session_id}}
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return await self.rag_application.ainvoke(question, config=config)
    
    def stream_query(self, question: str, session_id: str = None):
        """Stream the response to a query with session support and optimized configuration."""
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        # Generate session_id if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        # Create lightweight config with session_id for streaming
        config = {
            "configurable": {
                "session_id": session_id,
                "streaming_mode": True,  # Signal for optimized streaming
                "skip_expensive_operations": True  # Skip non-essential operations
            }
        }
        
        # Create initial state
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        # Stream with optimized session config
        return self.rag_application.graph.stream(initial_state, config=config, stream_mode="messages")
    
    async def astream_query(self, question: str):
        """Async version of stream_query."""
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return await self.rag_application.astream(question)
    
    def search_documents(self, query: str, k: int = 4):
        """
        Search for relevant documents without generating a response.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        if not self.is_initialized:
            raise ValueError("Application not initialized. Call initialize() first.")
        
        return global_retriever.get_relevant_documents(query, k=k)
    
    def get_stats(self) -> dict:
        """Get statistics about the RAG system."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        # Get collection info if available
        try:
            collection = self.vector_db.vector_database._collection
            doc_count = collection.count()
        except:
            doc_count = "unknown"
        
        return {
            "status": "initialized",
            "document_count": doc_count,
            "retriever_initialized": global_retriever.is_initialized(),
            "vector_db_type": "chroma"
        }

class RAGApplicationFactory:

    _implementations = {
        "rag_agent": RAGAgent,
        "rag_chain": RAGChain,
    }

    @classmethod
    def create_app(cls, app_type: str, llm: LLM, **kwargs) -> RAGApplication:
        if app_type not in cls._implementations:
            raise ValueError(f"Invalid app type: {app_type}")
        rag_application = cls._implementations[app_type]
        return rag_application(llm, **kwargs)

# Global application instance
app = RAGApplication()

def get_app() -> RAGApplication:
    """Get the global application instance."""
    return app 