from typing import Dict, List, Optional
from langchain_core.tools import tool
from utils.retriever import global_retriever

@tool
def retrieve_information(query: str) -> str:
    """
    Search the knowledge base for information related to the query.
    
    Args:
        query: The question or search term about the knowledge base.
        
    Returns:
        str: Relevant information from the knowledge base.
    """
    # Get relevant documents
    docs = global_retriever.get_relevant_documents(query, k=1)  # Only 1 doc
    
    if not docs:
        return "I couldn't find specific information about that in the knowledge base."
    
    # 🔧 VERY AGGRESSIVE TRUNCATION
    content = docs[0].page_content
    max_chars = 200  # ~50 tokens only
    
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    
    return f"Relevant information: {content}"