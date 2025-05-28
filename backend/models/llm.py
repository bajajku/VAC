from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
import transformers
from langchain.chat_models import init_chat_model
import os
import getpass


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
    """OpenAI/OpenRouter implementation.
    For DeepSeek OpenSource models.
    """
    
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
    
    def __init__(self, model_name: str, huggingface_api_token: str = None, quantization: bool = True, **kwargs):
        # Set default config for HF Pipeline
        default_config = {
            'temperature': 0.7,
            'max_length': 512,
            'do_sample': True
        }
        default_config.update(kwargs)
        super().__init__(model_name, **default_config)
        self.quantization = quantization
        self.huggingface_api_token = huggingface_api_token
        self.model, self.tokenizer = self.create_model()
        self.stopping_criteria = self.create_stopping_criteria()
        self.pipeline = self.create_pipeline(self.model, self.tokenizer)

    def create_model(self):
        if self.quantization:
            from torch import bfloat16
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )

            model_config = transformers.AutoConfig.from_pretrained(self.model_name, use_auth_token=self.huggingface_api_token)

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map='auto',
                use_auth_token=self.huggingface_api_token
            )

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.huggingface_api_token
            )

            return model, tokenizer
            
    def create_stopping_criteria(self):
        from utils.stopping_criteria import StopOnTokens
        from transformers import StoppingCriteriaList

        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.tokenizer)])
        return stopping_criteria



    def create_pipeline(self, model, tokenizer):
        streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_text_pipeline = transformers.pipeline(
            model=model,
            streamer=streamer,
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            stopping_criteria=self.stopping_criteria,  # without this model rambles during chat
            temperature=self.config['temperature'] if 'temperature' in self.config else 0.7,
            max_new_tokens=self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 512,
            repetition_penalty=self.config['repetition_penalty'] if 'repetition_penalty' in self.config else 1.1
        )   
        return generate_text_pipeline
    
    def create_llm(self):
        return HuggingFacePipeline(pipeline=self.pipeline)


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
            temperature=self.config['temperature'] if 'temperature' in self.config else 0.7,
            max_new_tokens=self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 512
        )


class MistralAILLM(BaseLLM):
    """Mistral AI implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def create_llm(self):
        model = init_chat_model(self.model_name,
                                model_provider="mistralai",
                                temperature=self.config['temperature'] if 'temperature' in self.config else 0.7,
                                max_tokens=self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 512,
                                timeout=10,
                                max_retries=3)
        return model

class ChatOpenAILLM(BaseLLM):
    """ChatOpenAI implementation."""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.api_key = api_key
        super().__init__(model_name, **kwargs)

    def create_llm(self):
        return ChatOpenAI(
            base_url=self.config['base_url'] if 'base_url' in self.config else "https://api.together.xyz/v1",
            model=self.model_name,
            api_key=self.api_key,
            **{k: v for k, v in self.config.items() if k != 'model'}
        )

class LLMFactory:
    """Factory class to create different LLM implementations."""
    
    _implementations = {
        'openai': OpenAILLM,
        'openrouter': OpenAILLM,  # Same as OpenAI but with different base_url
        'huggingface_pipeline': HuggingFacePipelineLLM,
        'huggingface_endpoint': HuggingFaceEndpointLLM,
        'mistralai': MistralAILLM,
        'chatopenai': ChatOpenAILLM,
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
            provider (str): The LLM provider ('openai', 'openrouter', 'huggingface_pipeline', 'huggingface_endpoint', 'mistralai', 'chatopenai')
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
