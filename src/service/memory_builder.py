from openai import AzureOpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage

from src.agents.aarma import AssistantAnswerRefinedMemoryAgent
from src.agents.rrma import ResidualRefinedMemoryAgent
from src.agents.uqrma import UserQuestionRefinedMemoryAgent
from src.group_chat.termination_strategy import TerminationStrategy
from src.group_chat.selection_strategy import SelectionStrategy
from src.memory.azure_ai_search import AzureAISearch
from src.memory.model import Memory, ResidualMemory, UserQuestionMemory, AssistantResponseMemory

class MemoryBuilder:
    def __init__(self, model_client: AzureOpenAIChatCompletionClient, search_client: AzureAISearch,
                 embedding_client: AzureOpenAI, embedding_model: str):
        self.search_client = search_client
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.model_client = model_client
        self.all_agents = self._build_agents()
        self.group_chat = self._build_group_chat(self.all_agents)

    def _build_agents(self) -> list[AssistantAgent | CodeExecutorAgent | UserProxyAgent]:
        init_agent = UserProxyAgent(
            name="init",
            description="Initial agent to start the conversation",
        )
        assistant_answer_refined_memory_agent = AssistantAnswerRefinedMemoryAgent(model_client=self.model_client).get_agent()
        residual_refined_memory_agent = ResidualRefinedMemoryAgent(model_client=self.model_client).get_agent()
        user_question_refined_memory_agent = UserQuestionRefinedMemoryAgent(model_client=self.model_client).get_agent()
        return [init_agent, assistant_answer_refined_memory_agent, residual_refined_memory_agent, user_question_refined_memory_agent]

    def _build_group_chat(self, participants: list[AssistantAgent | CodeExecutorAgent | UserProxyAgent]) -> SelectorGroupChat:
        return SelectorGroupChat(
            participants=participants,
            model_client=self.model_client,
            termination_condition=TerminationStrategy().get_termination_strategy(),
            selector_func=SelectionStrategy.state_transition,
        )

    async def _reset_all_agents(self, all_agents: list[AssistantAgent | CodeExecutorAgent | UserProxyAgent]) -> None:
        for agent in all_agents:
            await agent.on_reset(cancellation_token=None)
    
    def encode_text(self, text: str) -> list[float]:
        return self.embedding_client.embeddings.create(input=text, model=self.embedding_model).data[0].embedding
    
    async def build_memory(self, conversation: list[AgentEvent | ChatMessage | TaskResult | TextMessage ], user: str) -> list[Memory]:
        if len(conversation) < 2:
            return []
        conversation_str = ""
        memories = []
        for event in conversation:
            if isinstance(event, TextMessage):
                if event.source.upper() == "USER":
                    conversation_str += f"User: {event.content}\n"
                elif event.source.upper() == "ASSISTANT":
                    conversation_str += f"Assistant: {event.content}\n"
                else:
                    raise ValueError(f"Unknown source {event.source}")
            elif isinstance(event, ChatMessage):
                conversation_str += f"User: {event.message}\n"
            elif isinstance(event, AgentEvent):
                conversation_str += f"Assistant: {event.message}\n"
            elif isinstance(event, TaskResult):
                conversation_str += f"Assistant: {event.message}\n"
        
        memory_builder_response_stream = self.group_chat.run_stream(task=conversation_str)
        async for msg in memory_builder_response_stream:
            if isinstance(msg, TaskResult):
                continue
            agent = msg.source
            memory = msg.content
            print(f"Agent: {agent}, Memory: {memory}")
            if agent == ResidualRefinedMemoryAgent.name:
                memory = ResidualMemory(memory=memory, user=user)
            elif agent == UserQuestionRefinedMemoryAgent.name:
                memory = UserQuestionMemory(memory=memory, user=user)
            elif agent == AssistantAnswerRefinedMemoryAgent.name:
                memory = AssistantResponseMemory(memory=memory, user=user)
            elif agent == "user":
                continue
            else:
                raise ValueError(f"Unknown agent {agent}")
            if len(conversation) > 2:
                memories.append(memory)
            else:
                memories = [memory] if isinstance(memory, AssistantResponseMemory) else memories
        await self._reset_all_agents(self.all_agents)
        return memories

    async def persist_memory(self, memories: list[Memory]) -> None:
        memories_with_embeddings = []
        for memory in memories:
            if not memory.memoryVector:
                memory.memoryVector = self.encode_text(memory.memory)
            memories_with_embeddings.append(memory)
        await self.search_client.upload_memories(memories=memories_with_embeddings)
