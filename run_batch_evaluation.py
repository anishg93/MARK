import os
import json
import uuid
from dotenv import load_dotenv
from argparse import ArgumentParser

from openai import AzureOpenAI

from src.evaluation.info_cap_score import InformationCaptureScore
from src.data.model import EvaluationData

load_dotenv(override=True)

az_chat_completion_client = AzureOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_BASE_URL'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['AZURE_OPENAI_EVALUATION_API_VERSION'],
)
az_embedding_client = AzureOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_BASE_URL'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['AZURE_OPENAI_EMBEDDING_API_VERSION'],
)
info_cap_score = InformationCaptureScore()

def persist_evaluation(answers: list[EvaluationData], summary: dict[str, float], file_path: str = ".evaluation_output_data"):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    unique_id =str(uuid.uuid4())
    evaluation_result_file = os.path.join(file_path, f"{unique_id}_full.jsonl")
    with open(evaluation_result_file, "w") as f:
        for answer in answers:
            f.write(json.dumps(answer.to_dict()))
            f.write("\n")
    evaluation_summary_file = os.path.join(file_path, f"{unique_id}_summary.json")
    with open(evaluation_summary_file, "w") as f:
        f.write(json.dumps(summary))
    print(f"\n\n**** Evaluation results saved to {evaluation_result_file} and {evaluation_summary_file}")

def run_evaluation(answers: list[EvaluationData]):
    info_cap_score.set_weights(key_point_cov_score_weight=0.5, info_cov_score_weight=0.5)
    answers_with_scores = info_cap_score.evaluate(
        answers=answers,
        openai_client=az_chat_completion_client,
        embedding_client=az_embedding_client,
        openai_model=os.environ['AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME'],
        embedding_model=os.environ['AZURE_OPENAI_EMBEDDING_MODEL'],
        key_point_cs_threshold=0.8,
        info_cov_cs_threshold=0.9
    )
    print(info_cap_score.get_score())
    return answers_with_scores

def load_data(file_path: str) -> list[EvaluationData]:
    answers = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        for line in f:
            answer = json.loads(line)
            evaluation_data = EvaluationData(question=answer["question"],
                                            generated_answer=answer["generated_answer"],
                                            expected_answer=answer["expected_answer"] if "expected_answer" in answer else None,
                                            session_id=answer["session_id"] if "session_id" in answer else None,
                                            turn_id=answer["turn_id"]) if "turn_id" in answer else None
            answers.append(evaluation_data)
    return answers

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the file containing the answers.")
    args = parser.parse_args()
    file = args.file
    answers = load_data(file)
    answers_with_scores = run_evaluation(answers=answers)
    persist_evaluation(answers=answers_with_scores, summary=info_cap_score.get_score())
