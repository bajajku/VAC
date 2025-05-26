from langchain_chroma import Chroma
from langchain_core.documents import Document

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
                self.vector_database = Chroma(embedding_model=embedding_model, \
                                              persist_directory=kwargs.get("persist_directory") or "./chroma_db", \
                                              collection_name=kwargs.get("collection_name") or "default")
            case _:
                raise ValueError(f"Invalid vector database type: {type}")

    def add_documents(self, documents: list[Document]):
        self.vector_database.add_documents(documents)

