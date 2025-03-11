import re
import numpy as np
from openai import AzureOpenAI

from .base import EvaluationMetric
from src.data.model import EvaluationData

class InformationCoverageScore(EvaluationMetric):
    def __init__(self):
        super().__init__("InformationCoverageScore")
    
    def evaluate(self, **kwargs) -> list[EvaluationData]:
        score = 0.0
        answers_with_scores = []
        answers : list[EvaluationData] = kwargs.get("answers")
        cosine_similarity_threshold = kwargs.get("cosine_similarity_threshold")
        embedding_client = kwargs.get("embedding_client")
        model = kwargs.get("model")
        if not answers or not cosine_similarity_threshold or not embedding_client or not model:
            raise ValueError("Answers, cosine similarity threshold, embedding client, and model are required.")
        for answer in answers:
            similarity_scores = []
            info_coverage_scores = []
            generated = answer.generated_answer
            expected = answer.expected_answer
            if generated and expected:
                generated_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', generated)
                expected_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', expected)
                generated_vectors = self._convert_string_to_vector(generated_sentences, embedding_client, model)
                expected_vectors = self._convert_string_to_vector(expected_sentences, embedding_client, model)
                similarity_scores = [[float(self._calculate_cosine_similarity(generated_vector, expected_vector)) for expected_vector in expected_vectors] for generated_vector in generated_vectors]
                for i in range(len(similarity_scores)):
                    x = [sim for sim in similarity_scores[i] if sim >= np.quantile(similarity_scores[i], cosine_similarity_threshold)]
                    info_coverage_scores.append(len(x))
                info_coverage_score = float(np.mean([i / len(expected_sentences) for i in info_coverage_scores]))
            else: 
                info_coverage_score = 0.0
            answer.set_in_cov_cs_score(info_coverage_score)
            answers_with_scores.append(answer)
        if answers:
            score = sum([answer.in_cov_cs_score for answer in answers]) / len(answers)
        self.set_score(score)
        print(f"Information Coverage Score: {score} from {len(answers)} answers.")
        return answers_with_scores
    
    def _convert_string_to_vector(self, sentences: list[str], embedding_client: AzureOpenAI, model: str) -> list[float]:
        return [embedding_client.embeddings.create(input=sentence, model=model).data[0].embedding for sentence in sentences]
    
    def _calculate_cosine_similarity(self, generated: list[float], expected: list[float]) -> float:
        dot_product = sum([a * b for a, b in zip(generated, expected)])
        magnitude_generated = sum([a ** 2 for a in generated]) ** 0.5
        magnitude_expected = sum([b ** 2 for b in expected]) ** 0.5
        similarity = dot_product / (magnitude_generated * magnitude_expected)
        return similarity
