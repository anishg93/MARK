import os
import uuid
import asyncio
import json
import pandas as pd
from dotenv import load_dotenv
from argparse import ArgumentParser

from openai import AzureOpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage

from src.agents.cba import ChatbotAgent
from src.data.med_mcqa import MedMCQADataSet
from src.data.model import EvaluationData
from src.memory.azure_ai_search import AzureAISearch
from src.service.memory_builder import MemoryBuilder

import warnings; warnings.simplefilter('ignore')

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
print("================Initialization Complete===============")

def get_memory(query: str, top: int = 3, threshold: float = 0.01) -> str:
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

async def run_agent(eval_data: EvaluationData) -> EvaluationData:
    query = eval_data.question
    chatbot_agent = ChatbotAgent(model_client=az_openai_model_client, use_memory=USE_MEMORY).get_agent()
    if USE_MEMORY:
        memory = get_memory(query)
        query = f"## Memories:{memory}\n\n## Question:\n{query}"
    query_message = TextMessage(content=query, source="User")
    message_history = [query_message]
    response_stream = chatbot_agent.on_messages_stream(messages=message_history, cancellation_token=None)
    async for msg in response_stream:
        if isinstance(msg, Response):
            message_content = msg.chat_message.content
            eval_data.generated_answer = message_content
            eval_data.prompt_token_count = msg.chat_message.models_usage.prompt_tokens
            eval_data.completion_token_count = msg.chat_message.models_usage.completion_tokens
            return eval_data
        else:
            print(f"*** [AGENT_ERROR]: {msg}")
            return eval_data

def load_data(file_path: str, limit: int = 10, type: str = "med_mcqa") -> list[EvaluationData]:
    data = []
    if type == "med_mcqa":
        dataset = MedMCQADataSet(file_path)
        dataset.process_data(limit=limit)
        data = dataset.get_data()
        return [EvaluationData(
            question=d["question"],
            expected_answer=d["expected_answer"],
            session_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
        ) for d in data]
    elif type == "exp_2":
        dataset = pd.read_csv(file_path)
        dataset = dataset[dataset["human_eval"] == "Incorrect"]
        dataset = dataset.reset_index(drop=True)
        evaluation_data = []
        for i, row in dataset.iterrows():
            if limit and i >= limit:
                break
            evaluation_data.append(EvaluationData(
                question=row["question"],
                expected_answer=row["expected_answer"],
                session_id=row["session_id"],
                turn_id=row["turn_id"],
            ))
        return evaluation_data

def persist_experiment(answers: list[EvaluationData], file_path: str = ".evaluation_input_data"):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    unique_id =str(uuid.uuid4())
    experiment_result_file = os.path.join(file_path, f"{unique_id}.jsonl")
    with open(experiment_result_file, "w") as f:
        for answer in answers:
            f.write(json.dumps(answer.to_dict()))
            f.write("\n")
    print(f"\n\n**** Experiment results saved to - {experiment_result_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the file containing the answers.")
    parser.add_argument("--limit", type=int, help="Limit the number of records to read.", default=10)
    parser.add_argument("--type", type=str, default="med_mcqa", help="Type of dataset to read.", choices=["med_mcqa", "exp_2"])
    args = parser.parse_args()
    file = args.file
    limit = args.limit
    type = args.type
    answers = load_data(file_path=file, limit=limit, type=type)
    print(f"Loaded {len(answers)} records from {file}.")
    evaluation_results = []
    print("Started processing ", end="")
    for answer in answers:
        try:
            print(".", end="")
            evaluation_results.append(asyncio.run(run_agent(answer)))
            if len(evaluation_results) % 10 == 0:
                print(f"\nProcessed {len(evaluation_results)} records.\nStarted processing ", end="")
        except Exception as e:
            print(f"Error processing answer: {e}")
            continue
    persist_experiment(answers=evaluation_results)
