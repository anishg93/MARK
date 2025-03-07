import os
import uuid
import asyncio
import json
from dotenv import load_dotenv
from argparse import ArgumentParser

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage

from src.agents.cba import ChatbotAgent
from src.data.med_mcqa import MedMCQADataSet
from src.data.model import EvaluationData

import warnings; warnings.simplefilter('ignore')

load_dotenv(override=True)

print("================Starting Initialization================")
az_openai_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    model=os.environ['AZURE_OPENAI_MODEL_NAME'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_BASE_URL'],
    api_key=os.environ['AZURE_OPENAI_API_KEY']
)
chatbot_agent = ChatbotAgent(model_client=az_openai_model_client).get_agent()
print("================Initialization Complete===============")

async def run_agent(eval_data: EvaluationData) -> EvaluationData:
    query_message = TextMessage(content=eval_data.question, source="User")
    message_history = [query_message]
    response_stream = chatbot_agent.on_messages_stream(messages=message_history, cancellation_token=None)
    async for msg in response_stream:
        if isinstance(msg, Response):
            message_content = msg.chat_message.content
            eval_data.generated_answer = message_content
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
    return [EvaluationData(question=d["question"], expected_answer=d["expected_answer"]) for d in data]

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
    parser.add_argument("--type", type=str, default="med_mcqa", help="Type of dataset to read.", choices=["med_mcqa"])
    args = parser.parse_args()
    file = args.file
    limit = args.limit
    type = args.type
    answers = load_data(file_path=file, limit=limit, type=type)
    evaluation_results = []
    for answer in answers:
        evaluation_results.append(asyncio.run(run_agent(answer)))
    persist_experiment(answers=evaluation_results)
