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
        high_similarity_count = 0
        for answer in answers:
            actual = answer.actual_answer
            expected = answer.expected_answer
            if actual and expected:
                actual_vector = self._convert_string_to_vector(actual, embedding_client, model)
                expected_vector = self._convert_string_to_vector(expected, embedding_client, model)
                similarity = self._calculate_cosine_similarity(actual_vector, expected_vector)
                if similarity > cosine_similarity_threshold:
                    high_similarity_count += 1
            else:
                similarity = 0.0
            answer.set_in_cov_cs_score(similarity)
            answers_with_scores.append(answer)
        if answers:
            score = high_similarity_count / len(answers)
        self.set_score(score)
        print(f"Information Coverage Score: {score} from {len(answers)} answers.")
        return answers_with_scores
    
    def _convert_string_to_vector(self, string: str, embedding_client: AzureOpenAI, model: str) -> list[float]:
        return embedding_client.embeddings.create(input=string, model=model).data[0].embedding
    
    def _calculate_cosine_similarity(self, actual: list[float], expected: list[float]) -> float:
        dot_product = sum([a * b for a, b in zip(actual, expected)])
        magnitude_actual = sum([a ** 2 for a in actual]) ** 0.5
        magnitude_expected = sum([b ** 2 for b in expected]) ** 0.5
        similarity = dot_product / (magnitude_actual * magnitude_expected)
        return similarity
