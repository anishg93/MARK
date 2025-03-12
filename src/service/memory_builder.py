import json
from openai import AzureOpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage

from src.agents.aarma import AssistantAnswerRefinedMemoryAgent
from src.agents.rrma import ResidualRefinedMemoryAgent
from src.agents.uqrma import UserQuestionRefinedMemoryAgent
from src.memory.azure_ai_search import AzureAISearch
from src.memory.model import Memory, ResidualMemory, UserQuestionMemory, AssistantResponseMemory

class MemoryBuilder:
    def __init__(self, model_client: AzureOpenAIChatCompletionClient, search_client: AzureAISearch,
                 embedding_client: AzureOpenAI, embedding_model: str):
        self.search_client = search_client
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.model_client = model_client
        self.assistant_answer_refined_memory_agent = AssistantAnswerRefinedMemoryAgent(model_client=self.model_client).get_agent()
        self.residual_refined_memory_agent = ResidualRefinedMemoryAgent(model_client=self.model_client).get_agent()
        self.user_question_refined_memory_agent = UserQuestionRefinedMemoryAgent(model_client=self.model_client).get_agent()

    async def _reset_all_agents(self) -> None:
        await self.assistant_answer_refined_memory_agent.on_reset(cancellation_token=None)
        await self.residual_refined_memory_agent.on_reset(cancellation_token=None)
        await self.user_question_refined_memory_agent.on_reset(cancellation_token=None)
    
    def encode_text(self, text: str) -> list[float]:
        return self.embedding_client.embeddings.create(input=text, model=self.embedding_model).data[0].embedding
    
    async def build_memory(self, conversation: list[AgentEvent | ChatMessage | TaskResult | TextMessage ], user: str, agent: str) -> list[Memory]:
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
        conversation_message = TextMessage(content=conversation_str, source="User")
        
        res_response_stream = self.residual_refined_memory_agent.on_messages_stream(messages=[conversation_message], cancellation_token=None)
        async for msg in res_response_stream:
            memory_txt = msg.chat_message.content
            memory_txt = memory_txt.replace("{{", "{").replace("}}", "}")
            print(f"Residual Memory: {memory_txt}")
            try:
                for memory_value in json.loads(memory_txt)["residual_memory"]:
                    memory = ResidualMemory(memory=memory_value, user=user, agent=agent)
                    memories.append(memory)
            except Exception as e:
                print(f"Error parsing memory: {e}")
                memory = ResidualMemory(memory=memory_txt, user=user, agent=agent)
                memories.append(memory)
        
        user_response_stream = self.user_question_refined_memory_agent.on_messages_stream(messages=[conversation_message], cancellation_token=None)
        async for msg in user_response_stream:
            memory_txt = msg.chat_message.content
            memory_txt = memory_txt.replace("{{", "{").replace("}}", "}")
            print(f"User Question Memory: {memory_txt}")
            try:
                for memory_value in json.loads(memory_txt)["key_facts_about_user"]:
                    memory = UserQuestionMemory(memory=memory_value, user=user, agent=agent)
                    memories.append(memory)
            except Exception as e:
                print(f"Error parsing memory: {e}")
                memory = UserQuestionMemory(memory=memory_txt, user=user, agent=agent)
                memories.append(memory)
        
        assistant_response_stream = self.assistant_answer_refined_memory_agent.on_messages_stream(messages=[conversation_message], cancellation_token=None)
        async for msg in assistant_response_stream:
            memory_txt = msg.chat_message.content
            memory_txt = memory_txt.replace("{{", "{").replace("}}", "}")
            print(f"Assistant Response Memory: {memory_txt}")
            try:
                for memory_value in json.loads(memory_txt)["key_criteria"]:
                    memory = AssistantResponseMemory(memory=memory_value, user=user, agent=agent)
                    memories.append(memory)
            except Exception as e:
                print(f"Error parsing memory: {e}")
                memory = AssistantResponseMemory(memory=memory_txt, user=user, agent=agent)
                memories.append(memory)
    
        if len(conversation) == 2:
            tmp_memories = []
            for memory in memories:
                if isinstance(memory, AssistantResponseMemory):
                    tmp_memories.append(memory)
            memories = tmp_memories
        await self._reset_all_agents()
        return memories

    async def persist_memory(self, memories: list[Memory]) -> None:
        memories_with_embeddings = []
        for memory in memories:
            if not memory.memoryVector:
                memory.memoryVector = self.encode_text(memory.memory)
            memories_with_embeddings.append(memory)
        await self.search_client.upload_memories(memories=memories_with_embeddings)
