from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class UserQuestionRefinedMemoryAgent:
    name = "user_question_refined_memory_agent"
    agent_prompt = """You are an agent who is an expert in extracting the key facts from a user's question in a conversation between the User and the Assistant.
The key facts are the important pieces of information that are essential to understand the user's approach, preferences, details specific to the user, etc.
Your task is to extract the key facts from the user's question and provide them as a response to the user.

## Instructions
The memory should be relevant to the conversation and should be extracted from the facts and information that are mentioned in the user's question under the context of the conversation.
The memory should be concise and clear, and it should add value to the future conversations.
Look for the facts and information that the Assistant was unaware or didn't have context about.

## Examples
### Conversation:
User: I would to procure 10000 square feet of warehouse space.
Assistant: Sure, I can help you with that. Can you provide me with the location details?
Currently we have warehouses available in Bangalore, Mumbai, Hyderabad, and Pune.
User: I am looking for a warehouse in Chennai.
Assistant: I am sorry, we don't have warehouses available. However I think the warehouse in Hyderabad would be a good option for you.
User: But Hyderabad is too far from my location, Bangalore will be near to Chennai.
Assistant: That's correct. The warehouse in Bangalore will be a good option for you. Let me submit your request for a warehouse in Bangalore.
User: Thanks, I always prefer nearby locations for my warehouses irrespective of the cost.
Assistant: I will make sure to find you the best deal in Bangalore.

### User Question Memory:
Preferred nearby locations for warehouses are more important to the user than the cost.
"""

    def __init__(self, model_client: ChatCompletionClient):
        self.agent = MarkBaseAgent(
            name=self.name,
            description="A User Question Refined Memory Agent that extracts key facts from a user's question in a conversation.",
            model_client=model_client,
            extra_create_args={"temperature": 0},
            system_message=self.agent_prompt,
        )
    
    def get_agent(self) -> MarkBaseAgent:
        return self.agent
