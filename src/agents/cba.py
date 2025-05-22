from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class ChatbotAgent:
    name = "Assistant"
    agent_prompt = """You are an helpful assistant that can provide information and answer questions.
    You will be given a question and 4 answer choices and you need to select the best answer from the choices.
    """
    agent_prompt_with_memory = """
You are a highly specialized assistant designed to provide accurate and contextually relevant answers by leveraging multiple memory sources.  
Your task is to analyze a question and four provided answer choices, then select the most appropriate answer while ensuring alignment with stored user expectations, 
domain-specific knowledge, and prior refinements. You will be provided with three memory sources to refine and enhance your response, ensuring consistency, precision, and relevance.

Memory Sources & Prioritization Order:
## LLM Response Memory (General knowledge & prior response validation):
 - Captures key criteria that led to the User’s acceptance of previous responses.
 - Ensures continuity in response format, factual correctness, and domain alignment.

## Residual Memory (Implicit knowledge & user-specific refinements):
 - Stores information that refines the assistant’s understanding of terminology, accuracy, and user expectations.
 - Captures implied details, inferred relationships, and response corrections made during past interactions.

## User Question Memory (Intent alignment & phrasing consistency):
 - Retains user-provided facts, abbreviations, and domain-specific language for improved intent understanding.
 - Helps maintain consistency in how the assistant interprets similar queries.
"""

    def __init__(self, model_client: ChatCompletionClient, use_memory: bool = False):
        self.agent = MarkBaseAgent(
            name=self.name,
            description="Chatbot assistant",
            model_client=model_client,
            extra_create_args={"temperature": 0},
            system_message=self.agent_prompt if not use_memory else self.agent_prompt_with_memory,
        )
    
    def get_agent(self) -> MarkBaseAgent:
        return self.agent
