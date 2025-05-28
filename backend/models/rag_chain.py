from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from models.llm import LLM
from langchain.schema import BaseRetriever
from utils.prompt import Prompt
    
class RAGChain:
    def __init__(self, llm: LLM, retriever: BaseRetriever, prompt: Prompt):
        self.retriever = retriever
        self.llm = llm.create_chat()
        self.chain = self.create_chain()
        self.doc_chain = self.create_doc_chain()
        self.rag_chain = self.create_rag_chain()
        self.prompt = prompt
    
    def create_chain(self, chain_type: str = "rag"):
        match chain_type:
            case "rag":
                return self.create_rag_chain()
            case _:
                raise ValueError(f"Invalid chain type: {chain_type}")

    def create_doc_chain(self):
        return create_stuff_documents_chain(self.llm, self.prompt.get_prompt())

    def create_rag_chain(self):
        return create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=self.doc_chain,
        )