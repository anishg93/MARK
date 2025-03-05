from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class AssistantAnswerRefinedMemoryAgent:
    name = "assistant_answer_refined_memory_agent"
    agent_prompt = """You are an agent who is an expert in extracting the key criteria on why this answer was accepted by the user in a conversation between the User and the Assistant.
These key criteria are the memory that is essential to understand the user's approach, preferences, details specific to the user, etc.
Your task is to extract the key criteria from the Assistant's answer as memory and provide them as a response to the user.

## Instructions
The memory should very precise and focused on different aspects of the answer that made it acceptable to the user.
The memory should be concise and clear, and it should add value to the future conversations.

## Examples
### Conversation:
User: I would to procure 10000 square feet of warehouse space.
Assistant: Sure, I can help you with that. Can you provide me with the location details?
Currently we have warehouses available in Bangalore, Mumbai, Hyderabad, and Pune.
User: I am looking for a warehouse in Chennai.
Assistant: I am sorry, we don't have warehouses available. However I think the warehouse in Hyderabad would be a good option for you.
User: Don't say sorry. But Hyderabad is too far from my location, Bangalore will be near to Chennai.
Assistant: That's correct. The warehouse in Bangalore will be a good option for you. Let me submit your request for a warehouse in Bangalore.
User: Thanks, I always prefer nearby locations for my warehouses irrespective of the cost.
Assistant: I will make sure to find you the best deal in Bangalore.

### Assistant Answer Memory:
{{
    "key_criteria": ["accuracy", "location", "cost"],
}}
"""

    def __init__(self, model_client: ChatCompletionClient):
        self.agent = MarkBaseAgent(
            name=self.name,
            description="An Assistant Answer Refined Memory Agent that extracts key criteria from the Assistant's answer in a conversation.",
            model_client=model_client,
            extra_create_args={"temperature": 0},
            system_message=self.agent_prompt,
        )
    
    def get_agent(self) -> MarkBaseAgent:
        return self.agent
