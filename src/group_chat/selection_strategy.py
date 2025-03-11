from typing import Sequence
from autogen_agentchat.messages import AgentEvent, ChatMessage

from src.agents.aarma import AssistantAnswerRefinedMemoryAgent
from src.agents.rrma import ResidualRefinedMemoryAgent
from src.agents.uqrma import UserQuestionRefinedMemoryAgent

class SelectionStrategy:
    def __init__(self):
        pass

    @staticmethod
    def state_transition(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        last_speaker = messages[-1].source
        
        if last_speaker != AssistantAnswerRefinedMemoryAgent.name and last_speaker != UserQuestionRefinedMemoryAgent.name and last_speaker != ResidualRefinedMemoryAgent.name:
            return UserQuestionRefinedMemoryAgent.name
        elif last_speaker == UserQuestionRefinedMemoryAgent.name:
            return AssistantAnswerRefinedMemoryAgent.name
        elif last_speaker == AssistantAnswerRefinedMemoryAgent.name:
            return ResidualRefinedMemoryAgent.name
        elif last_speaker == ResidualRefinedMemoryAgent.name:
            return None
        else:
            return None
