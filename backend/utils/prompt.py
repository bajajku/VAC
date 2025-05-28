from langchain.prompts import PromptTemplate

class Prompt:
    def __init__(self, template: str):
        self.template = template

    def get_prompt(self):
        return PromptTemplate.from_template(self.template)
