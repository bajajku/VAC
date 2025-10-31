from models.llm import LLM
from typing import List

#TODO: Generate test cases for the RAG Evaluation system, using llm
class TestCaseGenerator:
    """Generate test cases for the RAG system.
    
    Args:
        llm: LLM model to use for generating test cases
        input_prompt: Prompt to use for generating test cases
        conditions: Conditions to use for generating test cases, can include evaluation criteria
    
    """

    def __init__(self, llm: LLM, input_prompt: str, conditions: List[str]):
        self.llm = llm
        self.input_prompt = input_prompt
        self.conditions = conditions
        self.test_cases = []

    
    def generate_test_cases(self) -> List[str]:
        
        # TODO: Implement the logic to generate test cases
        # self.test_cases = self.llm.generate_test_cases(self.input_prompt, self.conditions)
        return self.test_cases


    

