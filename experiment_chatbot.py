import os
from dotenv import load_dotenv

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import TextInput

from openai import AzureOpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core.models import RequestUsage

from src.agents.cba import ChatbotAgent
from src.persistence.database_setup import DataPersistence
from src.memory.azure_ai_search import AzureAISearch
from src.service.memory_builder import MemoryBuilder

load_dotenv(override=True)

# Assistant Type: To use the assistant with memory, set this to True, otherwise set it to False.
# If set to False, the assistant will use the memory builder to build memory and persist it.
# If set to True, the assistant will use the memory builder to retrieve memory and use it in the conversation, it will not build or persist memory.
USE_MEMORY = True

print("================Starting Initialization================")
az_openai_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    model=os.environ['AZURE_OPENAI_MODEL_NAME'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_BASE_URL'],
    api_key=os.environ['AZURE_OPENAI_API_KEY']
)
az_embedding_client = AzureOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_BASE_URL'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['AZURE_OPENAI_EMBEDDING_API_VERSION'],
)
az_embedding_model = os.environ['AZURE_OPENAI_EMBEDDING_MODEL']
az_ai_search_client = AzureAISearch(
    endpoint=os.environ['AZURE_SEARCH_ENDPOINT'],
    key=os.environ['AZURE_SEARCH_API_KEY'],
    index_name=os.environ['AZURE_SEARCH_INDEX_NAME']
)
az_ai_search_client.create_index(model=az_embedding_client)
memory_builder = MemoryBuilder(
    model_client=az_openai_model_client,
    search_client=az_ai_search_client,
    embedding_client=az_embedding_client,
    embedding_model=az_embedding_model
)
data_persistence = DataPersistence(enable_storage_provider=False)
print("================Initialization Complete===============")

def get_memory(query: str, top: int = 2, threshold: float = 0.02) -> str:
    query_vector = memory_builder.encode_text(query)
    residual_memories = az_ai_search_client.search_memory(
        query=query,
        query_vector=query_vector,
        type="residual",
        top=top,
        relevance_threshold=threshold
    )
    user_question_memories = az_ai_search_client.search_memory(
        query=query,
        query_vector=query_vector,
        type="user_question",
        top=top,
        relevance_threshold=threshold
    )
    assistant_memories = az_ai_search_client.search_memory(
        query=query,
        query_vector=query_vector,
        type="assistant_response",
        top=top,
        relevance_threshold=threshold
    )
    return f"""
{memory_builder.get_memory_string("LLM Response", assistant_memories)}
{memory_builder.get_memory_string("Residual", residual_memories)}
{memory_builder.get_memory_string("User Question", user_question_memories)}
    """

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

@cl.on_settings_update
async def setting_update(settings):
    cl.context.session.chat_settings = settings

@cl.on_chat_start
async def start_chat():
    print("=================Chat started================")
    await cl.ChatSettings([
        TextInput(id="experiment", label="Experiment", placeholder="Mention Experiment Name", initial="Experiment", tooltip="Name of the experiment")
    ]).send()
    cl.user_session.set("message_history", [])
    cl.user_session.set("total_usage", RequestUsage(prompt_tokens=0, completion_tokens=0))

async def run_agent(query: str):
    chatbot_agent = ChatbotAgent(model_client=az_openai_model_client, use_memory=USE_MEMORY).get_agent()
    if USE_MEMORY:
        memory = get_memory(query)
        print(f"=======> Memory: {memory}")
        query = f"{query}\n{memory}"
    query_message = TextMessage(content=query, source="User")
    total_usage = cl.user_session.get("total_usage")
    message_history = cl.user_session.get("message_history")
    message_history.append(query_message)
    response_stream = chatbot_agent.on_messages_stream(messages=message_history, cancellation_token=None)
    async for msg in response_stream:
        print(f"=======> Agent response: {msg}")
        if isinstance(msg, Response):
            message_content = msg.chat_message.content
            message_metadata = {
                "experiment": cl.context.session.chat_settings.get("experiment", None)
            }
            if msg.chat_message.models_usage:
                message_metadata["completion_tokens"] = msg.chat_message.models_usage.completion_tokens
                message_metadata["prompt_tokens"] = msg.chat_message.models_usage.prompt_tokens
                total_usage.completion_tokens += msg.chat_message.models_usage.completion_tokens
                total_usage.prompt_tokens += msg.chat_message.models_usage.prompt_tokens
            cl_msg = cl.Message(content=message_content, author=msg.chat_message.source, metadata=message_metadata)
            await cl_msg.send()
        message_history.append(msg.chat_message)
        print(f"=======> Message history: {message_history}")

@cl.on_message  # type: ignore
async def chat(message: cl.Message):
    await run_agent(message.content)  # type: ignore

@cl.on_chat_end  # type: ignore
async def end_chat():
    # Total Usage
    total_usage = cl.user_session.get("total_usage")
    message_metadata = {
        "experiment": cl.context.session.chat_settings.get("experiment", None),
        "total_completion_tokens": total_usage.completion_tokens,
        "total_prompt_tokens": total_usage.prompt_tokens,
        "total_tokens": total_usage.completion_tokens + total_usage.prompt_tokens,
    }
    message_content = f"Chat is terminated. You can start a new chat.\n"
    message_content += f"Total completion tokens: {total_usage.completion_tokens}\n"
    message_content += f"Total prompt tokens: {total_usage.prompt_tokens}\n"
    message_content += f"Total tokens: {total_usage.completion_tokens + total_usage.prompt_tokens}"
    cl_msg = cl.Message(content=message_content, author="termination", metadata=message_metadata)
    await cl_msg.send()
    
    # Memory Builder
    if not USE_MEMORY:
        memory_builder_response = await memory_builder.build_memory(conversation=cl.user_session.get("message_history"), user="user1", agent="medical_chatbot")
        memory_builder_string = "\n".join([memory.type + ": " + memory.memory for memory in memory_builder_response])
        cl_msg = cl.Message(content="Memory Builder Response:\n" + memory_builder_string,
                            author="memory", metadata={"experiment": cl.context.session.chat_settings.get("experiment", None)})
        await cl_msg.send()
        # Persist Memory
        await memory_builder.persist_memory(memory_builder_response)
        
    # Reset
    cl.user_session.set("message_history", [])
    cl.user_session.set("total_usage", RequestUsage(prompt_tokens=0, completion_tokens=0))
    
    print("=================Chat ended==================")

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Drug Elimination",
            message="A patient is administered 200 mg of a drug. 75 mg of the drug is eliminated from the body in 90 minutes. If the drug follows first order kinetics, how much drug will remain after 6 hours? a) 12.5 mg. b) 25 mg. c) 30 mg. d) 50 mg."
        ),
        cl.Starter(
            label="Extraradicular Microorganism",
            message="Most common extraradicular microorganism is / are: a) Actinomyces species. b) Propioni bacterium. c) Propionicum. d) All of the above."
        ),
        cl.Starter(
            label="Bone Sounding",
            message="Bone sounding done in modern times is performed by which method? a) Probing. b) CBCT. c) Radiovisiography. d) RVG."
        ),
        cl.Starter(
            label="Basal Cell Carcinoma",
            message="Which of the following is the most common site for the occurrence of a basal cell carcinoma? a) Buccal mucosa. b) Hard palate. c) Skin of the lower lip. d) Dorsum of the tongue."
        ),
        cl.Starter(
            label="Dental Implant",
            message="Which of the following is not a contraindication for dental implants? a) Uncontrolled diabetes. b) Smoking. c) Osteoporosis. d) Hypertension."
        )
    ]
