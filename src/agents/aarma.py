from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class AssistantAnswerRefinedMemoryAgent:
    name = "assistant_answer_refined_memory_agent"
    agent_prompt = """You are an expert memory extraction agent responsible for capturing key criteria that led to the User's acceptance of the Assistant’s response in a conversation.

## Objective:
Your goal is to identify and store critical memory points that reflect the User's preferences, thought process, constraints, and expectations for future interactions.

## Instructions:
- Extract Key Criteria: Identify essential factors in the Assistant’s response that influenced the User’s acceptance, agreement, or decision-making.
- Capture User-Specific Preferences: Look for implicit and explicit patterns in user behavior, such as preferred formats, accuracy expectations, or domain-specific details.
- Ensure Multi-Turn Continuity: Retain information that enhances coherence and recall across multiple turns, enabling the Assistant to provide more personalized and relevant responses in the future.
- Maximize Utilization: Prioritize facts that will significantly impact future interactions rather than redundant or generic details.
- Be Precise and Concise: The extracted memory should be structured, easy to reference, and directly applicable in subsequent conversations.

## Examples
### Conversation:
User: What is the concentration of tropicamide:  
a) 0.01  
b) 0.02  
c) 0.03  
d) 0.04  

Assistant: Tropicamide is typically available in 0.5% and 1% solutions for ophthalmic use. None of the provided options (0.01, 0.02, 0.03, 0.04) match the commonly used concentrations.  

User: I believe the concentration used is 0.5%-1%, which translates to 0.005-0.01 in decimal form.  

Assistant: You're correct. The ophthalmic concentration of tropicamide is 0.5%-1%, which is equivalent to 0.005-0.01 in decimal form.  

User: Which of the given options is correct?  

Assistant: Based on the available information, the closest correct option is a) 0.01.  

### Extracted Memory:

{{
    "key_criteria": [
        "Precise numerical representation improves user acceptance. Responses should use decimal values (0.005-0.01) when discussing concentrations.",
        "User prefers definitive answers over broad explanations. Responses should aim to validate choices rather than providing general information.",
        "Clarifications and corrections enhance trust. Acknowledging and refining earlier statements contributes to response acceptance.",
        "The user values expert-backed confirmations. Future responses should emphasize accuracy and professional validation."
    ]
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
