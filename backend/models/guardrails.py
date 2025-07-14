from pydantic import BaseModel
from typing import List
from guardrails import Guard

'''
Guardrails is a class that contains a list of guards.
So we have multiple guards, and we can select which guards to use at each node in graph.

'''

class Guardrails(BaseModel):
    
    guards: List[Guard]
