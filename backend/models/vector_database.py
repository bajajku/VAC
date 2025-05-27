from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from scripts.data_collection.json_parser import JsonParser

'''
    Vector Database class to create and manage vector databases.
    Currently supports Chroma.
'''

class VectorDatabase:
    def __init__(self):
        self.vector_database = None

    def create_vector_database(self, embedding_model, type: str = "chroma", **kwargs):
        match type:
            case "chroma":
                self.vector_database = Chroma(embedding_function=HuggingFaceEmbeddings(model_name=embedding_model), \
                                              persist_directory=kwargs.get("persist_directory") or "./chroma_db", \
                                              collection_name=kwargs.get("collection_name") or "default")
            case _:
                raise ValueError(f"Invalid vector database type: {type}")

    def add_documents(self, documents: list[Document]):
        self.vector_database.add_documents(documents)

    def load_documents_from_json(self, json_file_path: str):
        """
        Load documents from a JSON file using JsonParser and add them to the vector database.
        
        Args:
            json_file_path (str): Path to the JSON file containing crawled data
        """
        if self.vector_database is None:
            raise ValueError("Vector database not initialized. Call create_vector_database() first.")
        
        # Use JsonParser to parse the JSON file
        parser = JsonParser(json_file_path)
        parser.parse_json()
        
        # Add the parsed documents to the vector database
        self.add_documents(parser.documents)
        
        return len(parser.documents)

if __name__ == "__main__":
    vector_database = VectorDatabase()
    vector_database.create_vector_database(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    parser = JsonParser("/Users/kunalbajaj/VAC/backend/scripts/data_collection/crawl_results/crawl_results_20250526_133954.json")
    parser.parse_json()
    vector_database.add_documents(parser.documents)