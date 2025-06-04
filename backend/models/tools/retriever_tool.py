from typing import Dict, List, Optional
from langchain_core.tools import tool
from utils.retriever import global_retriever

@tool
def retrieve_information(query: str, max_results: int = 5, 
                        retrieval_strategy: str = "hybrid",
                        use_reranking: bool = True) -> str:
    """
    Search the knowledge base for information related to the query using advanced retrieval techniques.
    
    Args:
        query: The question or search term about the knowledge base.
        max_results: Maximum number of documents to retrieve (default: 5).
        retrieval_strategy: Retrieval strategy to use ('similarity', 'mmr', 'hybrid', 'ensemble').
        use_reranking: Whether to enable advanced re-ranking of results.
        
    Returns:
        str: Relevant information from the knowledge base with enhanced processing.
    """
    if not global_retriever.is_initialized():
        return "The knowledge base is not initialized. Please initialize the vector database first."
    
    try:
        # Update retriever configuration for this query
        if hasattr(global_retriever._retriever, 'update_config'):
            global_retriever._retriever.update_config(
                max_results=max_results,
                retrieval_strategy=retrieval_strategy,
                enable_reranking=use_reranking
            )
        
        # Get relevant documents using enhanced retrieval
        docs = global_retriever.get_relevant_documents(query, k=max_results)
        
        if not docs:
            return "I couldn't find specific information about that in the knowledge base."
        
        # Process and format the results with better content handling
        formatted_results = []
        total_chars = 0
        max_total_chars = 2000  # Increased from 200 to 2000 for better context
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            
            # Create a rich result entry with metadata
            result_entry = f"**Source {i}**"
            
            # Add source information if available
            if 'source' in metadata:
                source = metadata['source']
                # Truncate long URLs/paths for readability
                if len(source) > 50:
                    source = "..." + source[-47:]
                result_entry += f" (from: {source})"
            
            # Add content type and relevance info if available
            if 'content_type' in metadata:
                result_entry += f" [Type: {metadata['content_type']}]"
            
            if 'retriever_sources' in metadata:
                result_entry += f" [Retrieved via: {', '.join(metadata['retriever_sources'])}]"
            
            result_entry += ":\n"
            
            # Smart content truncation based on available space
            remaining_chars = max_total_chars - total_chars - len(result_entry)
            
            if remaining_chars > 100:  # Only include if we have reasonable space
                # Intelligent truncation - try to keep complete sentences
                if len(content) <= remaining_chars:
                    truncated_content = content
                else:
                    # Try to find a good breaking point (sentence end)
                    truncate_at = remaining_chars - 20  # Leave some buffer
                    
                    # Look for sentence endings near the truncation point
                    sentence_ends = ['.', '!', '?', '\n']
                    best_break = truncate_at
                    
                    for end_char in sentence_ends:
                        last_occurrence = content.rfind(end_char, 0, truncate_at)
                        if last_occurrence > best_break * 0.7:  # Don't break too early
                            best_break = last_occurrence + 1
                            break
                    
                    truncated_content = content[:best_break].strip()
                    if best_break < len(content):
                        truncated_content += "..."
                
                result_entry += truncated_content
                total_chars += len(result_entry)
                formatted_results.append(result_entry)
                
                # Stop if we're approaching the limit
                if total_chars >= max_total_chars * 0.9:
                    break
            else:
                # If we're running out of space, just mention there are more results
                if i < len(docs):
                    formatted_results.append(f"... and {len(docs) - i + 1} more relevant sources found.")
                break
        
        # Combine all results
        if formatted_results:
            result = "Relevant information found:\n\n" + "\n\n".join(formatted_results)
            
            # Add summary statistics
            result += f"\n\n📊 Retrieved {len(docs)} documents"
            if hasattr(global_retriever._retriever, 'retrieval_strategy'):
                result += f" using {global_retriever._retriever.retrieval_strategy} strategy"
            
            return result
        else:
            return "Found relevant documents but couldn't format them properly. Please try with a more specific query."
    
    except Exception as e:
        return f"Error during retrieval: {str(e)}. Please try again or contact support."

@tool
def get_retrieval_stats(query: str) -> str:
    """
    Get detailed statistics about retrieval performance for a given query.
    
    Args:
        query: The query to analyze
        
    Returns:
        str: Retrieval statistics and performance metrics
    """
    if not global_retriever.is_initialized():
        return "The knowledge base is not initialized."
    
    try:
        if hasattr(global_retriever._retriever, 'get_retrieval_stats'):
            stats = global_retriever._retriever.get_retrieval_stats(query)
            
            formatted_stats = "🔍 **Retrieval Statistics**\n\n"
            for key, value in stats.items():
                if isinstance(value, list):
                    formatted_stats += f"**{key.replace('_', ' ').title()}**: {', '.join(value)}\n"
                else:
                    formatted_stats += f"**{key.replace('_', ' ').title()}**: {value}\n"
            
            return formatted_stats
        else:
            return "Retrieval statistics not available with current retriever configuration."
    
    except Exception as e:
        return f"Error getting retrieval stats: {str(e)}"

@tool 
def configure_retriever(retrieval_strategy: str = "hybrid",
                       similarity_threshold: float = 0.7,
                       max_results: int = 10,
                       enable_reranking: bool = True) -> str:
    """
    Configure the retriever with specific parameters for optimal performance.
    
    Args:
        retrieval_strategy: Strategy to use ('similarity', 'mmr', 'hybrid', 'ensemble')
        similarity_threshold: Minimum similarity score threshold (0.0-1.0)
        max_results: Maximum number of results to return
        enable_reranking: Whether to enable advanced re-ranking
        
    Returns:
        str: Configuration status
    """
    if not global_retriever.is_initialized():
        return "The knowledge base is not initialized."
    
    try:
        if hasattr(global_retriever._retriever, 'update_config'):
            global_retriever._retriever.update_config(
                retrieval_strategy=retrieval_strategy,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                enable_reranking=enable_reranking
            )
            
            return f"""✅ **Retriever Configuration Updated**

**Strategy**: {retrieval_strategy}
**Similarity Threshold**: {similarity_threshold}
**Max Results**: {max_results}
**Re-ranking Enabled**: {enable_reranking}

The retriever has been configured with these settings for improved performance."""
        else:
            return "Retriever configuration not supported with current setup."
    
    except Exception as e:
        return f"Error configuring retriever: {str(e)}"