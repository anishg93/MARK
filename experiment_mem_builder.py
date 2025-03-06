import os
from dotenv import load_dotenv

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import TextInput

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.base import TaskResult
from autogen_core.models import RequestUsage

from src.agents.aarma import AssistantAnswerRefinedMemoryAgent
from src.agents.rrma import ResidualRefinedMemoryAgent
from src.agents.uqrma import UserQuestionRefinedMemoryAgent
from src.group_chat.termination_strategy import TerminationStrategy
from src.group_chat.selection_strategy import SelectionStrategy
from src.persistence.database_setup import DataPersistence
from src.customization.actions import CustomActions

load_dotenv(override=True)

print("================Starting Initialization================")
az_openai_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    model=os.environ['AZURE_OPENAI_MODEL_NAME'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_BASE_URL'],
    api_key=os.environ['AZURE_OPENAI_API_KEY']
)

data_persistence = DataPersistence(enable_storage_provider=False)

init_agent = UserProxyAgent(
    name="init",
    description="Initial agent to start the conversation",
)
assistant_answer_refined_memory_agent = AssistantAnswerRefinedMemoryAgent(model_client=az_openai_model_client).get_agent()
residual_refined_memory_agent = ResidualRefinedMemoryAgent(model_client=az_openai_model_client).get_agent()
user_question_refined_memory_agent = UserQuestionRefinedMemoryAgent(model_client=az_openai_model_client).get_agent()
all_agents = [init_agent, assistant_answer_refined_memory_agent, residual_refined_memory_agent, user_question_refined_memory_agent]
memory_builder_group_chat = SelectorGroupChat(
    participants=all_agents,
    model_client=az_openai_model_client,
    termination_condition=TerminationStrategy().get_termination_strategy(),
    selector_func=SelectionStrategy.state_transition,
)
print("================Initialization Complete===============")

async def reset_all_agents(all_agents: list[AssistantAgent | CodeExecutorAgent | UserProxyAgent]) -> None:
    for agent in all_agents:
        await agent.on_reset(cancellation_token=None)

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=data_persistence.get_connection_url_async(), storage_provider=data_persistence.get_storage_provider())

@cl.password_auth_callback
def password_auth_callback(username: str, password: str) -> bool:
    if (username == os.environ['CHAINLIT_USERNAME']) and (password == os.environ['CHAINLIT_PASSWORD']):
        return cl.User(
            identifier=username,
            metadata={"role": os.environ['CHAINLIT_ROLE'], "provider": "credentials"}
        )
    return None

@cl.action_callback("thumbs_up_action")
async def thumbs_up_action_callback(action: cl.Action) -> None:
    await CustomActions.thumbs_up_down_action_handler(
        message_id=action.payload["message_id"], thread_id = action.payload["thread_id"], value=action.payload["value"], data_layer=get_data_layer())

@cl.action_callback("thumbs_down_action")
async def thumbs_down_action_callback(action: cl.Action) -> None:
    await CustomActions.thumbs_up_down_action_handler(
        message_id=action.payload["message_id"], thread_id = action.payload["thread_id"], value=action.payload["value"], data_layer=get_data_layer())

@cl.on_settings_update
async def setting_update(settings):
    cl.context.session.chat_settings = settings

@cl.on_chat_start
async def start_chat():
    print("=================Chat started================")
    await cl.ChatSettings([
        TextInput(id="experiment", label="Experiment", placeholder="Mention Experiment Name", initial="Experiment 1", tooltip="Name of the experiment")
    ]).send()

async def run_team(query: str):
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    response_stream = memory_builder_group_chat.run_stream(task=query)
    async for msg in response_stream:
        if hasattr(msg, "content"):
            message_content = f"Agent {msg.source.upper()}:\n{msg.content}"
            message_metadata = {
                "experiment": cl.context.session.chat_settings.get("experiment", None)
            }
            if msg.models_usage:
                message_metadata["completion_tokens"] = msg.models_usage.completion_tokens
                message_metadata["prompt_tokens"] = msg.models_usage.prompt_tokens
                total_usage.completion_tokens += msg.models_usage.completion_tokens
                total_usage.prompt_tokens += msg.models_usage.prompt_tokens
            cl_msg = cl.Message(content=message_content, author=msg.source, metadata=message_metadata)
            sent_message = await cl_msg.send()
            current_thread_id = cl.context.current_run.thread_id
            custom_actions = CustomActions(message_id=sent_message.id, thread_id=current_thread_id)
            cl_msg.actions = [custom_actions.get_thumbs_up_action(), custom_actions.get_thumbs_down_action()]
            await cl_msg.update()

        if isinstance(msg, TaskResult):
            message_metadata = {
                "experiment": cl.context.session.chat_settings.get("experiment", None),
                "total_completion_tokens": total_usage.completion_tokens,
                "total_prompt_tokens": total_usage.prompt_tokens,
                "total_tokens": total_usage.completion_tokens + total_usage.prompt_tokens,
            }
            message_content = f"Termination condition met. Team and Agents are reset. You can start a new chat.\n"
            message_content += f"Total completion tokens: {total_usage.completion_tokens}\n"
            message_content += f"Total prompt tokens: {total_usage.prompt_tokens}\n"
            message_content += f"Total tokens: {total_usage.completion_tokens + total_usage.prompt_tokens}"
            cl_msg = cl.Message(content=message_content, author="termination", metadata=message_metadata)
            await reset_all_agents(all_agents)
            await cl_msg.send()

@cl.on_message  # type: ignore
async def chat(message: cl.Message):
    await run_team(message.content)  # type: ignore

@cl.on_chat_end  # type: ignore
async def end_chat():
    print("=================Chat ended==================")
    await reset_all_agents(all_agents)
