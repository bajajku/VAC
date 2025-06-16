import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import transformers
# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.stop_list = ['\nHuman:', '\n```\n']
        self.stop_token_ids = [tokenizer(x)['input_ids'] for x in self.stop_list]


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
