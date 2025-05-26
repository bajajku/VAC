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
    docs = global_retriever.get_relevant_documents(query, k=2)
    
    if not docs:
        return "I couldn't find specific information about that in the knowledge base."
    
    # Combine the content from relevant documents
    results = "\n\n".join([doc.page_content for doc in docs])
    return f"Found the following information in the knowledge base:\n\n{results}"