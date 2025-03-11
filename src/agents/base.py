from typing import List, Any, Optional, Sequence, AsyncGenerator, Mapping
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, UserMessage, AssistantMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage
from autogen_agentchat.base import Response

class MarkBaseAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        system_message: str,
        model_client: ChatCompletionClient,
        description: Optional[str] = None,
        extra_create_args: Mapping[str, Any] = {},
        **kwargs
    ):
        super().__init__(
            name=name,
            description=description,
            model_client=model_client,
            system_message=system_message,
            **kwargs
        )
        self._extra_create_args = extra_create_args
    
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")
    
    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        # Add messages to the model context.
        for msg in messages:
            await self._model_context.add_message(UserMessage(content=msg.content, source=msg.source))

        # Inner messages.
        inner_messages: List[AgentEvent | ChatMessage] = []

        # Generate an inference result based on the current model context.
        llm_messages = self._system_messages + await self._model_context.get_messages()
        result = await self._model_client.create(
            llm_messages, tools=self._tools + self._handoff_tools, cancellation_token=cancellation_token, extra_create_args=self._extra_create_args
        )
        response = result.content.strip()

        # Add the response to the model context.
        await self._model_context.add_message(AssistantMessage(content=response, source=self.name))

        # Check if the response is a string and return it.
        yield Response(
            chat_message=TextMessage(content=response, source=self.name, models_usage=result.usage),
            inner_messages=inner_messages,
        )
        return
