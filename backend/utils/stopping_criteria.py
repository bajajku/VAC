import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import transformers

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.stop_list = ['\nHuman:', '\n```\n']
        # Convert to tensors immediately and store device info
        self.stop_token_ids = []
        for stop_text in self.stop_list:
            try:
                token_ids = tokenizer(stop_text)['input_ids']
                # Convert to tensor for proper comparison
                self.stop_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
            except Exception as e:
                print(f"Warning: Failed to tokenize stop text '{stop_text}': {e}")
                continue

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_token_ids:
            return False
            
        try:
            for stop_ids_tensor in self.stop_token_ids:
                # Ensure stop_ids_tensor is on the same device as input_ids
                stop_ids_tensor = stop_ids_tensor.to(input_ids.device)
                
                # Get the length we need to compare
                stop_length = stop_ids_tensor.shape[0]
                if stop_length > input_ids.shape[1]:
                    continue  # Skip if stop sequence is longer than input
                
                # Extract the last tokens from input_ids for comparison
                last_tokens = input_ids[0][-stop_length:]
                
                # Compare tensors (both are now tensors of the same type and device)
                if torch.equal(last_tokens, stop_ids_tensor):
                    return True
        except Exception as e:
            # Don't break generation if stopping criteria fails
            print(f"Warning: Stopping criteria error: {e}")
            return False
            
        return False
