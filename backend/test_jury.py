from utils.helper import pretty_print_jury_response
from models.jury import Jury
import os
from dotenv import load_dotenv
from models.llm import LLM

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# llm = LLM(provider="chatopenai", model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=TOGETHER_API_KEY)


config = [{
    "provider": "chatopenai",
    "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "api_key": TOGETHER_API_KEY,
},
{
    "provider": "chatopenai",
    "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "api_key": TOGETHER_API_KEY,
},
{
    "provider": "chatopenai",
    "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "api_key": TOGETHER_API_KEY,
}
]

jury = Jury(llm_configs=config, max_workers=3)


pretty_print_jury_response(jury.deliberate("What is the the meaning of life?", return_individual_responses=True))








