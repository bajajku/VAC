'''
This module provides a fallback service for the RAG application.
It is used to handle errors and provide a fallback response when the RAG application fails.
'''

class FallbackService:
    def __init__(self):
        self.fallback_response = "I'm sorry, I'm not able to answer that question. Please try again."

    def get_fallback_response(self):
        return self.fallback_response
    
    def set_fallback_response(self, response: str):
        self.fallback_response = response
        