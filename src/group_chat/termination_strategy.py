from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.conditions import MaxMessageTermination, SourceMatchTermination

from src.agents.rrma import ResidualRefinedMemoryAgent

class TerminationStrategy:
    def __init__(self, max_messages: int = 15):
        self.max_messages = max_messages

    def get_termination_strategy(self) -> TerminationCondition:
        max_messages_termination = MaxMessageTermination(max_messages=self.max_messages)
        source_match_termination = SourceMatchTermination(sources=[ResidualRefinedMemoryAgent.name])
        termination = max_messages_termination | source_match_termination
        return termination
