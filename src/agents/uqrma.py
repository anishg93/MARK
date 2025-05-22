from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class UserQuestionRefinedMemoryAgent:
    name = "user_question_refined_memory_agent"
    agent_prompt = """You are an expert agent responsible for extracting key facts, abbreviations, and terminology from a User’s question in a conversation with the Assistant.
These elements are essential for understanding the User's intent, preferred phrasing, and domain-specific language.

## Instructions:
- Identify user-provided facts, abbreviations, and terminology that influence decision-making.
- Extract explicitly stated details while ensuring domain-specific terms and preferred phrasing are retained.
- Recognize patterns in the User’s terminology to improve future response alignment.
- Ensure the memory is concise, clear, and useful for future interactions.
- Focus on details the Assistant was unaware of or didn’t previously have context about.

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
    "user_question_refined_memory": [
        "User prefers decimal notation for medical concentrations. Future responses should convert percentage values (e.g., 0.5%) into decimals (e.g., 0.005).",
        "User relies on domain-specific abbreviations. Retain shorthand and technical terms where relevant.",
        "User expects a direct answer when listing multiple-choice options. The Assistant should avoid broad explanations when a clear choice is requested.",
        "User assumes ophthalmic concentration standards. Future responses should align with established medical dosage norms."
    ]
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
