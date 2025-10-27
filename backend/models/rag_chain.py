from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from models.llm import LLM
from langchain.schema import BaseRetriever
from utils.prompt import Prompt
from utils.retriever import global_retriever
from abc import ABC, abstractmethod
class Chain(ABC):
    def __init__(self, llm: LLM, **kwargs):
        self.llm = llm.create_chat()
        self.prompt = kwargs.get("prompt")
    
    @abstractmethod
    def create_chain(self):
        pass

    @abstractmethod
    def invoke(self, question: str) -> str:
        pass

    @abstractmethod
    def ainvoke(self, question: str) -> str:
        pass

class NormalChain(Chain):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(llm, **kwargs)
        self.chain = self.create_chain()
    
    def create_chain(self):
        chain = self.prompt.get_prompt() | self.llm
        return chain
    
    def invoke(self, question: str) -> str:
        return self.chain.invoke({"context": question})
    
    def ainvoke(self, question: str) -> str:
        return self.chain.ainvoke({"context": question})
    

class RAGChain(Chain):
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
    
    def invoke(self, question: str):
        response = self.rag_chain.invoke({"input": question})
        return response
    
    def ainvoke(self, question: str) -> str:
        return self.rag_chain.ainvoke({"input": question})
    
    def stream(self, question: str) -> str:
        return self.rag_chain.stream({"input": question})