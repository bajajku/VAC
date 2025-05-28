from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from models.llm import LLM
from langchain.schema import BaseRetriever
from utils.prompt import Prompt
from utils.retriever import global_retriever

class RAGChain:
    def __init__(self, llm: LLM, **kwargs):
        self.retriever = global_retriever._retriever.retriever
        self.llm = llm.create_chat()
        self.prompt = kwargs.get("prompt")
        self.doc_chain = self.create_doc_chain()
        self.rag_chain = self.create_rag_chain()
        self.chain = self.create_chain()
    
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
    
    def invoke(self, question: str) -> str:
        return self.rag_chain.invoke({"input": question})
    
    def ainvoke(self, question: str) -> str:
        return self.rag_chain.ainvoke({"input": question})
    
    def stream(self, question: str) -> str:
        return self.rag_chain.stream({"input": question})