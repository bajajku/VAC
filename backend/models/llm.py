from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint

'''
    LLM class to create and manage LLMs.
'''

#TODO: Add support for other LLM providers

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.client = self.create_llm()
    
    @abstractmethod
    def create_llm(self):
        """Create and return the LLM client."""
        pass
    
    def create_chat(self):
        """Return the chat client."""
        return self.client


class OpenAILLM(BaseLLM):
    """OpenAI/OpenRouter implementation."""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.api_key = api_key
        # Set default config for OpenAI
        default_config = {
            'temperature': 0.7,
            'streaming': True,
            'base_url': "https://openrouter.ai/api/v1"
        }
        default_config.update(kwargs)
        super().__init__(model_name, **default_config)
    
    def create_llm(self):
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            **{k: v for k, v in self.config.items() if k != 'model'}
        )


class HuggingFacePipelineLLM(BaseLLM):
    """HuggingFace Pipeline implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        # Set default config for HF Pipeline
        default_config = {
            'temperature': 0.7,
            'max_length': 512,
            'do_sample': True
        }
        default_config.update(kwargs)
        super().__init__(model_name, **default_config)
    
    def create_llm(self):
        return HuggingFacePipeline.from_model_id(
            model_id=self.model_name,
            task="text-generation",
            model_kwargs=self.config
        )


class HuggingFaceEndpointLLM(BaseLLM):
    """HuggingFace Endpoint implementation."""
    
    def __init__(self, model_name: str, huggingfacehub_api_token: str = None, **kwargs):
        self.api_token = huggingfacehub_api_token
        # Set default config for HF Endpoint
        default_config = {
            'temperature': 0.7,
            'max_new_tokens': 512
        }
        default_config.update(kwargs)
        super().__init__(model_name, **default_config)
    
    def create_llm(self):
        return HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_token,
            model_kwargs=self.config
        )


class LLMFactory:
    """Factory class to create different LLM implementations."""
    
    _implementations = {
        'openai': OpenAILLM,
        'openrouter': OpenAILLM,  # Same as OpenAI but with different base_url
        'huggingface_pipeline': HuggingFacePipelineLLM,
        'huggingface_endpoint': HuggingFaceEndpointLLM,
    }
    
    @classmethod
    def create_llm(cls, provider: str, model_name: str, **kwargs) -> BaseLLM:
        """
        Create an LLM instance based on the provider.
        
        Args:
            provider (str): The LLM provider ('openai', 'openrouter', 'huggingface_pipeline', 'huggingface_endpoint')
            model_name (str): The model identifier
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseLLM: An instance of the specified LLM implementation
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider not in cls._implementations:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(cls._implementations.keys())}")
        
        llm_class = cls._implementations[provider]
        return llm_class(model_name, **kwargs)
    
    @classmethod
    def register_implementation(cls, name: str, implementation_class):
        """Register a new LLM implementation."""
        cls._implementations[name] = implementation_class


# Convenience wrapper class that maintains your original interface
class LLM:
    """Wrapper class that provides a unified interface for different LLM providers."""
    
    def __init__(self, provider: str, model_name: str, **kwargs):
        """
        Initialize the LLM with a specific provider.
        
        Args:
            provider (str): The LLM provider ('openai', 'openrouter', 'huggingface_pipeline', 'huggingface_endpoint')
            model_name (str): The model identifier
            **kwargs: Additional configuration parameters specific to the provider
        """
        self.provider = provider
        self.model_name = model_name
        self.llm = LLMFactory.create_llm(provider, model_name, **kwargs)
    
    def create_chat(self):
        """Return the chat client."""
        return self.llm.create_chat()
    
    def switch_provider(self, new_provider: str, **kwargs):
        """Switch to a different provider while keeping the same model name."""
        self.provider = new_provider
        self.llm = LLMFactory.create_llm(new_provider, self.model_name, **kwargs)
