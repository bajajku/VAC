from typing import List, Dict, Any, Optional, Callable, Union
from models.llm import LLM, LLMFactory
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import Counter
import statistics
import re
import json


class Jury:
    """
    A jury of LLMs that can make ensemble decisions with a variable number of models.
    Supports different voting strategies and parallel execution for efficiency.
    """
    
    def __init__(self, llm_configs: List[Dict[str, Any]], voting_strategy: str = "majority", 
                 max_workers: int = None, **kwargs):
        """
        Initialize the jury with multiple LLM configurations.
        
        Args:
            llm_configs: List of dictionaries containing LLM configuration
                        Each dict should have 'provider', 'model_name', and optional kwargs
                        Example: [
                            {'provider': 'openai', 'model_name': 'gpt-3.5-turbo', 'api_key': 'key1'},
                            {'provider': 'openrouter', 'model_name': 'mistral-7b', 'api_key': 'key2'},
                            {'provider': 'huggingface_endpoint', 'model_name': 'microsoft/DialoGPT-medium'}
                        ]
            voting_strategy: Strategy for combining results ('majority', 'weighted', 'unanimous', 'first_valid')
            max_workers: Maximum number of parallel workers (defaults to number of LLMs)
            **kwargs: Additional configuration for all LLMs
        """
        self.llm_configs = llm_configs
        self.voting_strategy = voting_strategy
        self.max_workers = max_workers or len(llm_configs)
        self.global_kwargs = kwargs
        
        # Initialize all LLMs
        self.llms = self._initialize_llms()
        self.jury_size = len(self.llms)
        
        # Voting strategy mapping
        self.voting_strategies = {
            'majority': self._majority_vote,
            'weighted': self._weighted_vote,
            'unanimous': self._unanimous_vote,
            'first_valid': self._first_valid_vote,
            'average_score': self._average_score_vote,
            'consensus': self._consensus_vote
        }
        
        if voting_strategy not in self.voting_strategies:
            raise ValueError(f"Unsupported voting strategy: {voting_strategy}. "
                           f"Available: {list(self.voting_strategies.keys())}")
    
    def _initialize_llms(self) -> List[LLM]:
        """Initialize all LLM instances from configurations."""
        llms = []
        for i, config in enumerate(self.llm_configs):
            try:
                # Merge global kwargs with specific config
                merged_config = {**self.global_kwargs, **config}
                provider = merged_config.pop('provider')
                model_name = merged_config.pop('model_name')
                
                llm = LLM(provider=provider, model_name=model_name, **merged_config)
                llms.append(llm)
                print(f"✅ Initialized LLM {i+1}/{len(self.llm_configs)}: {provider}/{model_name}")
            except Exception as e:
                print(f"❌ Failed to initialize LLM {i+1}: {e}")
                continue
        
        if not llms:
            raise ValueError("No LLMs could be initialized successfully")
        
        return llms
    
    def add_llm(self, provider: str, model_name: str, **kwargs) -> None:
        """Add a new LLM to the jury."""
        try:
            merged_config = {**self.global_kwargs, **kwargs}
            llm = LLM(provider=provider, model_name=model_name, **merged_config)
            self.llms.append(llm)
            self.jury_size = len(self.llms)
            print(f"✅ Added LLM to jury: {provider}/{model_name}")
        except Exception as e:
            print(f"❌ Failed to add LLM: {e}")
    
    def remove_llm(self, index: int) -> None:
        """Remove an LLM from the jury by index."""
        if 0 <= index < len(self.llms):
            self.llms.pop(index)
            self.jury_size = len(self.llms)
            print(f"✅ Removed LLM at index {index}")
        else:
            raise IndexError(f"Invalid index {index}. Jury has {len(self.llms)} LLMs")
    
    def deliberate(self, prompt: str, return_individual_responses: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Have the jury deliberate on a prompt and return the consensus.
        
        Args:
            prompt: The prompt to send to all LLMs
            return_individual_responses: Whether to return individual responses
            
        Returns:
            Union[str, Dict]: Either the consensus response or detailed results
        """
        # Get responses from all LLMs in parallel
        individual_responses = self._get_parallel_responses(prompt)
        
        # Apply voting strategy
        consensus = self.voting_strategies[self.voting_strategy](individual_responses)
        
        if return_individual_responses:
            return {
                'consensus': consensus,
                'individual_responses': individual_responses,
                'jury_size': self.jury_size,
                'voting_strategy': self.voting_strategy,
                'response_count': len(individual_responses)
            }
        
        return consensus
    
    async def adeliberate(self, prompt: str, return_individual_responses: bool = False) -> Union[str, Dict[str, Any]]:
        """Async version of deliberate."""
        individual_responses = await self._get_async_responses(prompt)
        
        consensus = self.voting_strategies[self.voting_strategy](individual_responses)
        
        if return_individual_responses:
            return {
                'consensus': consensus,
                'individual_responses': individual_responses,
                'jury_size': self.jury_size,
                'voting_strategy': self.voting_strategy,
                'response_count': len(individual_responses)
            }
        
        return consensus
    
    def _get_parallel_responses(self, prompt: str) -> List[Dict[str, Any]]:
        """Get responses from all LLMs in parallel using ThreadPoolExecutor."""
        responses = []

        print("Yes we are here")
        
        def query_llm(llm_index: int, llm: LLM) -> Dict[str, Any]:
            try:
                chat = llm.create_chat()
                print("____________________")
                print(llm)
                print("____________________")
                print(type(chat))
                print("____________________")
                response = chat.invoke(prompt)
                print("____________________")
                print(f"RAW RESPONSE FROM {llm.provider}/{llm.model_name}:")
                print(response)
                print("____________________")
                
                # Extract content with better debugging
                if hasattr(response, 'content'):
                    response_content = response.content
                    print(f"CONTENT EXTRACTED: {response_content[:200]}...")
                else:
                    response_content = str(response)
                    print(f"STRING CONVERSION: {response_content[:200]}...")
                
                result = {
                    'index': llm_index,
                    'response': response_content,
                    'provider': llm.provider,
                    'model': llm.model_name,
                    'success': True,
                    'error': None
                }
                print(f"SUCCESS RESULT FOR {llm.provider}: {result['success']}")
                return result
            except Exception as e:
                print(f"❌ ERROR in {llm.provider}/{llm.model_name}: {e}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                
                return {
                    'index': llm_index,
                    'response': None,
                    'provider': llm.provider,
                    'model': llm.model_name,
                    'success': False,
                    'error': str(e)
                }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_llm = {
                executor.submit(query_llm, i, llm): i 
                for i, llm in enumerate(self.llms)
            }
            
            for future in as_completed(future_to_llm):
                response = future.result()
                responses.append(response)
                print(f"COLLECTED RESPONSE FROM {response['provider']}: success={response['success']}")
        
        # Sort by index to maintain order
        responses.sort(key=lambda x: x['index'])
        
        # Debug: Print summary of all responses
        print(f"FINAL RESPONSES SUMMARY:")
        for i, resp in enumerate(responses):
            print(f"  {i}: {resp['provider']}/{resp['model']} - success: {resp['success']}")
            if not resp['success']:
                print(f"      Error: {resp['error']}")
        
        return responses
    
    async def _get_async_responses(self, prompt: str) -> List[Dict[str, Any]]:
        """Get responses from all LLMs asynchronously."""
        async def query_llm_async(llm_index: int, llm: LLM) -> Dict[str, Any]:
            try:
                chat = llm.create_chat()
                response = await chat.ainvoke(prompt)
                return {
                    'index': llm_index,
                    'response': response.content if hasattr(response, 'content') else str(response),
                    'provider': llm.provider,
                    'model': llm.model_name,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                return {
                    'index': llm_index,
                    'response': None,
                    'provider': llm.provider,
                    'model': llm.model_name,
                    'success': False,
                    'error': str(e)
                }
        
        tasks = [query_llm_async(i, llm) for i, llm in enumerate(self.llms)]
        responses = await asyncio.gather(*tasks)
        return list(responses)
    
    # Voting strategies
    def _majority_vote(self, responses: List[Dict[str, Any]]) -> str:
        """Simple majority voting based on exact matches."""
        successful_responses = [r['response'] for r in responses if r['success'] and r['response']]
        
        if not successful_responses:
            return "No valid responses received from jury"
        
        # Count occurrences
        response_counts = Counter(successful_responses)
        most_common = response_counts.most_common(1)[0]
        
        return most_common[0]
    
    def _weighted_vote(self, responses: List[Dict[str, Any]]) -> str:
        """Weighted voting - can assign weights based on model reliability."""
        print(f"🗳️ WEIGHTED VOTE - Processing {len(responses)} responses:")
        for i, resp in enumerate(responses):
            print(f"  Response {i}: {resp['provider']}/{resp['model']} - success: {resp['success']}")
            if resp['success'] and resp['response']:
                print(f"    Content preview: {resp['response'][:100]}...")
            elif resp['success']:
                print(f"    Success but no content: {resp['response']}")
            else:
                print(f"    Failed: {resp['error']}")
        
        # Default equal weights, but you can customize this
        weights = {
            'openai': 1.2,
            'openrouter': 1.0,
            'huggingface_pipeline': 0.8,
            'huggingface_endpoint': 0.9,
            'mistralai': 1.1,
            'chatopenai': 1.0
        }
        
        weighted_responses = {}
        for response in responses:
            if response['success'] and response['response']:
                content = response['response']
                weight = weights.get(response['provider'], 1.0)
                print(f"  Adding weighted response from {response['provider']} (weight: {weight})")
                
                if content in weighted_responses:
                    weighted_responses[content] += weight
                else:
                    weighted_responses[content] = weight
        
        if not weighted_responses:
            print("  ❌ No valid weighted responses found!")
            return "No valid responses received from jury"
        
        result = max(weighted_responses, key=weighted_responses.get)
        print(f"  ✅ Weighted vote result: {len(weighted_responses)} unique responses processed")
        return result
    
    def _unanimous_vote(self, responses: List[Dict[str, Any]]) -> str:
        """Require unanimous agreement, otherwise return aggregated response."""
        successful_responses = [r['response'] for r in responses if r['success'] and r['response']]
        
        if not successful_responses:
            return "No valid responses received from jury"
        
        # For JSON responses (like evaluation scores), parse and compare semantically
        parsed_responses = []
        json_responses = []
        
        for response in successful_responses:
            try:
                # Try to extract and parse JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_json = json.loads(json_str)
                    parsed_responses.append(parsed_json)
                    json_responses.append(response)
                else:
                    # Not a JSON response, treat as plain text
                    parsed_responses.append(response)
                    json_responses.append(response)
            except (json.JSONDecodeError, AttributeError):
                # Failed to parse JSON, treat as plain text
                parsed_responses.append(response)
                json_responses.append(response)
        
        # Check for unanimous agreement
        if len(set(successful_responses)) == 1:
            # Exact string match - perfect unanimity
            return successful_responses[0]
        
        # For JSON responses, check if scores are unanimous even if formatting differs
        if all(isinstance(resp, dict) and 'score' in resp for resp in parsed_responses):
            scores = [resp['score'] for resp in parsed_responses]
            if len(set(scores)) == 1:
                # Unanimous score agreement - return the first response (they all have same score)
                return json_responses[0]
        
        # No unanimity - for evaluation tasks, return average score in JSON format
        if all(isinstance(resp, dict) and 'score' in resp for resp in parsed_responses):
            avg_score = statistics.mean([resp['score'] for resp in parsed_responses])
            combined_reasoning = " | ".join([resp.get('reasoning', '') for resp in parsed_responses])
            avg_confidence = statistics.mean([resp.get('confidence', 0.5) for resp in parsed_responses])
            
            return json.dumps({
                "score": round(avg_score, 1),
                "reasoning": f"No unanimous agreement. Average of {len(parsed_responses)} responses: {combined_reasoning}",
                "confidence": round(avg_confidence, 2)
            })
        
        # For non-JSON responses, return the original behavior
        return f"No unanimous agreement. Responses varied: {len(set(successful_responses))} different answers received."
    
    def _first_valid_vote(self, responses: List[Dict[str, Any]]) -> str:
        """Return the first valid response."""
        for response in responses:
            if response['success'] and response['response']:
                return response['response']
        
        return "No valid responses received from jury"
    
    def _average_score_vote(self, responses: List[Dict[str, Any]]) -> str:
        """For numerical responses, return the average. For text, use majority."""
        successful_responses = [r['response'] for r in responses if r['success'] and r['response']]
        
        if not successful_responses:
            return "No valid responses received from jury"
        
        # Try to extract numbers for averaging
        numbers = []
        for response in successful_responses:
            # Extract numbers from response
            nums = re.findall(r'-?\d+\.?\d*', response)
            if nums:
                try:
                    numbers.append(float(nums[0]))
                except ValueError:
                    continue
        
        if numbers:
            average = statistics.mean(numbers)
            return f"Average score: {average:.2f}"
        else:
            # Fall back to majority vote for non-numerical responses
            return self._majority_vote(responses)
    
    def _consensus_vote(self, responses: List[Dict[str, Any]]) -> str:
        """Build consensus by finding common themes."""
        successful_responses = [r['response'] for r in responses if r['success'] and r['response']]
        
        if not successful_responses:
            return "No valid responses received from jury"
        
        if len(successful_responses) == 1:
            return successful_responses[0]
        
        # Simple consensus building - combine unique responses
        unique_responses = list(set(successful_responses))
        if len(unique_responses) <= 3:
            return " | ".join(unique_responses)
        else:
            return self._majority_vote(responses)
    
    def get_jury_info(self) -> Dict[str, Any]:
        """Get information about the current jury composition."""
        return {
            'jury_size': self.jury_size,
            'voting_strategy': self.voting_strategy,
            'max_workers': self.max_workers,
            'llm_details': [
                {
                    'provider': llm.provider,
                    'model': llm.model_name,
                    'index': i
                }
                for i, llm in enumerate(self.llms)
            ]
        }
    
    def change_voting_strategy(self, new_strategy: str) -> None:
        """Change the voting strategy."""
        if new_strategy not in self.voting_strategies:
            raise ValueError(f"Unsupported voting strategy: {new_strategy}. "
                           f"Available: {list(self.voting_strategies.keys())}")
        
        self.voting_strategy = new_strategy
        print(f"✅ Changed voting strategy to: {new_strategy}")


# Factory function for easy jury creation
def create_jury(llm_configs: List[Dict[str, Any]], voting_strategy: str = "majority", **kwargs) -> Jury:
    """
    Factory function to create a jury with common configurations.
    
    Example usage:
        jury = create_jury([
            {'provider': 'openai', 'model_name': 'gpt-3.5-turbo'},
            {'provider': 'openrouter', 'model_name': 'mistral-7b-instruct'},
            {'provider': 'huggingface_endpoint', 'model_name': 'microsoft/DialoGPT-medium'}
        ], voting_strategy='majority')
    """
    return Jury(llm_configs, voting_strategy, **kwargs)


# Predefined jury configurations
class JuryPresets:
    """Common jury configurations for different use cases."""
    
    @staticmethod
    def diverse_jury(api_keys: Dict[str, str]) -> List[Dict[str, Any]]:
        """A diverse jury with different model types."""
        return [
            {'provider': 'openai', 'model_name': 'gpt-3.5-turbo', 'api_key': api_keys.get('openai')},
            {'provider': 'openrouter', 'model_name': 'mistral-7b-instruct', 'api_key': api_keys.get('openrouter')},
            {'provider': 'mistralai', 'model_name': 'mistral-tiny', 'api_key': api_keys.get('mistral')}
        ]
    
    @staticmethod
    def openai_jury(api_key: str) -> List[Dict[str, Any]]:
        """A jury of different OpenAI models."""
        return [
            {'provider': 'openai', 'model_name': 'gpt-3.5-turbo', 'api_key': api_key},
            {'provider': 'openai', 'model_name': 'gpt-4', 'api_key': api_key},
            {'provider': 'openai', 'model_name': 'gpt-4-turbo', 'api_key': api_key}
        ]
    
    @staticmethod
    def small_jury(api_keys: Dict[str, str]) -> List[Dict[str, Any]]:
        """A small, fast jury for quick decisions."""
        return [
            {'provider': 'openai', 'model_name': 'gpt-3.5-turbo', 'api_key': api_keys.get('openai')},
            {'provider': 'openrouter', 'model_name': 'mistral-7b-instruct', 'api_key': api_keys.get('openrouter')}
        ]
        