import json
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
import tqdm
from models.llm import LLM
from models.rag_chain import NormalChain
from utils.prompt import Prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

class DataCleaner:
    def __init__(self, unclean_data: json):
        self.unclean_data = unclean_data
        self.chain = None
        self.cleaned_documents = []
        self.data_cleaning_prompt = self._create_cleaning_prompt()

    def _create_cleaning_prompt(self) -> Prompt:
        """Create a prompt template for data cleaning."""
        template = """
You are a data cleaning specialist. Your task is to analyze the given document and clean it for relevance and quality.

Please perform the following tasks:
1. Remove any irrelevant information (ads, navigation elements, boilerplate text)
2. Don't change the original content, just clean it.
3. Fix any obvious formatting issues
4. Extract only the main content that would be useful for a knowledge base
5. Ensure the text is coherent and well-structured

Document to clean:
{context}

Please return only the cleaned, relevant content. If the document contains no useful information, return "NO_USEFUL_CONTENT".
"""
        return Prompt(template=template)

    def _initialize_chain(self):
        """Initialize the NormalChain for data cleaning."""
        if self.chain is None:
            # Get API key from environment
            api_key = os.environ.get('TOGETHER_API_KEY')
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set. Please set it to use the data cleaner.")
            
            llm = LLM(
                provider="chatopenai", 
                model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                api_key=api_key
            )
            self.chain = NormalChain(llm=llm, prompt=self.data_cleaning_prompt)
            print(self.chain)

    def clean_data(self) -> List[Document]:
        """
        Clean the unclean data and return a list of cleaned Document objects.
        
        Returns:
            List[Document]: List of cleaned documents in LangChain Document format
        """
        self._initialize_chain()
        
        # Process the JSON data
        if isinstance(self.unclean_data, dict):
            data_dict = self.unclean_data
        else:
            # If it's a JSON string, parse it
            data_dict = json.loads(self.unclean_data) if isinstance(self.unclean_data, str) else self.unclean_data
        
        cleaned_documents = []
        
        for key, item in tqdm.tqdm(data_dict.items(), desc="Cleaning documents"):
            try:
                # Extract content from the item
                if isinstance(item, dict):
                    # Assume structure like {'text_content': '...', 'title': '...', 'description': '...'}
                    content = item.get('text_content', '')
                    title = item.get('title', '')
                    description = item.get('description', '')
                    
                    # Combine content for cleaning
                    full_content = f"Title: {title}\nDescription: {description}\nContent: {content}"
                else:
                    full_content = str(item)
                
                # Skip if content is empty or too short
                if not full_content.strip() or len(full_content.strip()) < 10:
                    continue
                
                # Clean the content using the chain
                cleaned_content = self.chain.invoke({'context': full_content})
                
                cleaned_content = cleaned_content.content
                
                # Create metadata
                metadata = {
                    'source': key,
                    'original_title': title if isinstance(item, dict) else '',
                    'original_description': description if isinstance(item, dict) else '',
                    'cleaned': True,
                    'cleaning_method': 'llm_chain'
                }
                
                # Create Document object
                document = Document(
                    page_content=cleaned_content.strip(),
                    metadata=metadata
                )
                
                cleaned_documents.append(document)
                
            except Exception as e:
                print(f"Error processing item {key}: {str(e)}")
                continue
        
        self.cleaned_documents = cleaned_documents
        return cleaned_documents

    def get_cleaned_documents(self) -> List[Document]:
        """
        Get the cleaned documents. If not already cleaned, perform cleaning first.
        
        Returns:
            List[Document]: List of cleaned documents
        """
        if not self.cleaned_documents:
            return self.clean_data()
        return self.cleaned_documents

    def save_cleaned_documents(self, output_path: str):
        """
        Save cleaned documents to a JSON file.
        
        Args:
            output_path (str): Path to save the cleaned documents
        """
        if not self.cleaned_documents:
            self.clean_data()
        
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert documents to serializable format
        serializable_docs = []
        for doc in self.cleaned_documents:
            serializable_docs.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(self.cleaned_documents)} cleaned documents to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cleaning process.
        
        Returns:
            Dict[str, Any]: Statistics about original vs cleaned data
        """
        original_count = len(self.unclean_data) if isinstance(self.unclean_data, dict) else 0
        cleaned_count = len(self.cleaned_documents)
        
        return {
            'original_documents': original_count,
            'cleaned_documents': cleaned_count,
            'documents_removed': original_count - cleaned_count,
            'cleaning_efficiency': f"{(cleaned_count/original_count)*100:.1f}%" if original_count > 0 else "0%"
        }