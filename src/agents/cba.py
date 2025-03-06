from autogen_core.models import ChatCompletionClient
from .base import MarkBaseAgent

class ChatbotAgent:
    name = "Assistant"
    agent_prompt = """You are an helpful assistant that can provide information and answer questions."""

    def __init__(self, model_client: ChatCompletionClient):
        self.agent = MarkBaseAgent(
            name=self.name,
            description="Chatbot assistant",
            model_client=model_client,
            extra_create_args={"temperature": 0},
            system_message=self.agent_prompt,
        )
    
    def get_agent(self) -> MarkBaseAgent:
        return self.agent
