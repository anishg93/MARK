from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class UserQuestionRefinedMemoryAgent:
    name = "user_question_refined_memory_agent"
    agent_prompt = """You are an expert agent responsible for extracting key facts from a User’s question in a conversation with the Assistant.
These key facts represent critical details about the User’s approach, preferences, and specific requirements, which are essential for understanding their intent.

## Instructions:
- Identify the most important details from the User's question that influence decision-making.
- Extract explicit facts while considering the conversation’s context.
- Ensure the memory is concise, clear, and valuable for future interactions.
- Focus on facts the Assistant was unaware of or didn’t previously have context about.

## Examples
### Conversation:
User: Concentration of tropicamide: a) 0.01 b) 0.02 c) 0.03 d) 0.04.
Assistant: The concentration of tropicamide can vary depending on its intended use. However, commonly available concentrations of tropicamide for ophthalmic use are 0.5% and 1%. Therefore, none of the options provided (0.01, 0.02, 0.03, 0.04) are typical concentrations for tropicamide.
User: I think concentration used is 0.5-l% drops, is nothing but - 0.005-0.010.
Assistant: I apologize for any confusion. Yes, you are correct. The concentration of tropicamide commonly used for ophthalmic purposes is typically 0.5% to 1%. In decimal form, this would be 0.005 to 0.01.
User: Which will be the correct option?
Assistant: Based on the information provided, the correct option for the concentration of tropicamide would be: a) 0.01

### Extracted User Question Memory:
{{
"key_facts_about_user": ["User prefers decimal conversion of tropicamide concentration"]
}}
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
