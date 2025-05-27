import os
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = True

@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    type: str = "chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_db"
    collection_name: str = "rag_collection"

@dataclass
class RAGConfig:
    """Main RAG system configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval_k: int = 4
    debug: bool = False

class Settings:
    """Settings manager for the RAG application."""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> RAGConfig:
        """Load configuration from environment variables and defaults."""
        
        # LLM Configuration
        llm_config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "0")) or None,
            streaming=os.getenv("LLM_STREAMING", "true").lower() == "true"
        )
        
        # Vector DB Configuration
        vector_db_config = VectorDBConfig(
            type=os.getenv("VECTOR_DB_TYPE", "chroma"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=os.getenv("VECTOR_DB_PERSIST_DIR", "./chroma_db"),
            collection_name=os.getenv("VECTOR_DB_COLLECTION", "rag_collection")
        )
        
        # Main RAG Configuration
        return RAGConfig(
            llm=llm_config,
            vector_db=vector_db_config,
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def get_llm_kwargs(self) -> dict:
        """Get LLM configuration as kwargs."""
        llm_kwargs = {
            "temperature": self.config.llm.temperature,
            "streaming": self.config.llm.streaming,
        }
        
        if self.config.llm.api_key:
            llm_kwargs["api_key"] = self.config.llm.api_key
        
        if self.config.llm.base_url:
            llm_kwargs["base_url"] = self.config.llm.base_url
        
        if self.config.llm.max_tokens:
            llm_kwargs["max_tokens"] = self.config.llm.max_tokens
        
        return llm_kwargs
    
    def get_vector_db_kwargs(self) -> dict:
        """Get vector database configuration as kwargs."""
        return {
            "embedding_model": self.config.vector_db.embedding_model,
            "type": self.config.vector_db.type,
            "persist_directory": self.config.vector_db.persist_directory,
            "collection_name": self.config.vector_db.collection_name
        }
    
    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check API key for cloud providers
        if self.config.llm.provider in ["openai", "openrouter"] and not self.config.llm.api_key:
            issues.append(f"API key required for {self.config.llm.provider}")
        
        # Check embedding model format
        if "/" not in self.config.vector_db.embedding_model:
            issues.append("Embedding model should be in format 'organization/model-name'")
        
        return issues

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings 