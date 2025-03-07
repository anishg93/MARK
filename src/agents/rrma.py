from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class ResidualRefinedMemoryAgent:
    name = "residual_refined_memory_agent"
    agent_prompt = """You are an expert agent responsible for extracting residual memory from a conversation between the User and the Assistant.
Residual memory consists of implied or inferred information that was not explicitly stated but can be deduced from the conversation.

Instructions:
Identify key facts or relationships not directly mentioned but inferred from the conversation.
Focus on context that the Assistant was unaware of or did not explicitly acknowledge.
Ensure the extracted memory is concise, clear, and valuable for future interaction.

## Examples
### Conversation:
User: Concentration of tropicamide: a) 0.01 b) 0.02 c) 0.03 d) 0.04.
Assistant: The concentration of tropicamide can vary depending on its intended use. However, commonly available concentrations of tropicamide for ophthalmic use are 0.5% and 1%. Therefore, none of the options provided (0.01, 0.02, 0.03, 0.04) are typical concentrations for tropicamide.
User: I think concentration used is 0.5-l% drops, is nothing but - 0.005-0.010.
Assistant: I apologize for any confusion. Yes, you are correct. The concentration of tropicamide commonly used for ophthalmic purposes is typically 0.5% to 1%. In decimal form, this would be 0.005 to 0.01.
User: Which will be the correct option?
Assistant: Based on the information provided, the correct option for the concentration of tropicamide would be: a) 0.01

### Extracted Residual Memory:
{{
"residual_memory": ["Concentration of tropicamide in decimal form would be 0.005 to 0.01."]
}}
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
