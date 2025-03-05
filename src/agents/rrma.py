from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class ResidualRefinedMemoryAgent:
    name = "residual_refined_memory_agent"
    agent_prompt = """You are an agent who is an expert in extracting residual memory from a conversation between the User and the Assistant.
The residual memory is the information that is not explicitly mentioned in the conversation but is implied or can be inferred from the context.
Your task is to extract the residual memory from the conversation and provide it as a response to the user.

## Instructions
The memory should be relevant to the conversation and should be extracted from the facts and information that are mentioned in the conversation.
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
Assistant: That's correct. The warehouse in Bangalore will be a good option for you.

### Residual Memory:
The warehouse in Bangalore will be a good option for the uses from Chennai as it is near to Chennai.
"""

    def __init__(self, model_client: ChatCompletionClient):
        self.agent = MarkBaseAgent(
            name=self.name,
            description="A Residual Refined Memory Agent that extracts residual memory from a conversation.",
            model_client=model_client,
            extra_create_args={"temperature": 0},
            system_message=self.agent_prompt,
        )
    
    def get_agent(self) -> MarkBaseAgent:
        return self.agent
