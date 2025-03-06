from .base import EvaluationMetric
from .info_cov_score import InformationCoverageScore
from .key_point_cov_score import KeyPointCoverageScore
from src.data.model import EvaluationData

class InformationCaptureScore(EvaluationMetric):
    def __init__(self):
        super().__init__("InformationCaptureScore")
        self.score = 0.0
        self.key_point_cov_score_client = KeyPointCoverageScore()
        self.info_cov_score_client = InformationCoverageScore()
        self.weights = {
            "key_point_cov_score": 0.5,
            "info_cov_score": 0.5
        }
        self.info_cov_score = 0.0
        self.key_point_cov_score = 0.0
    
    def set_weights(self, key_point_cov_score_weight: float, info_cov_score_weight: float):
        self.weights["key_point_cov_score"] = key_point_cov_score_weight
        self.weights["info_cov_score"] = info_cov_score_weight
    
    def evaluate(self, **kwargs) -> list[EvaluationData]:
        answers : list[EvaluationData] = kwargs.get("answers")
        openai_client = kwargs.get("openai_client")
        embedding_client = kwargs.get("embedding_client")
        openai_model = kwargs.get("openai_model")
        embeddings_model = kwargs.get("embedding_model")
        key_point_cs_threshold = kwargs.get("key_point_cs_threshold")
        info_cov_cs_threshold = kwargs.get("info_cov_cs_threshold")
        
        if not answers or not openai_client or not embedding_client or not openai_model or not embeddings_model:
            raise ValueError("Answers, OpenAI client, embedding client, OpenAI model, and embeddings model are required.")
        
        if not key_point_cs_threshold:
            key_point_cs_threshold = 0.9
        if not info_cov_cs_threshold:
            info_cov_cs_threshold = 0.9
        
        answers_with_scores = self.key_point_cov_score_client.evaluate(
            answers=answers,
            openai_client=openai_client,
            embedding_client=embedding_client,
            openai_model=openai_model,
            embedding_model=embeddings_model,
            cosine_similarity_threshold=key_point_cs_threshold
        )
        self.key_point_cov_score = self.key_point_cov_score_client.score
        
        answers_with_scores = self.info_cov_score_client.evaluate(
            answers=answers_with_scores,
            cosine_similarity_threshold=info_cov_cs_threshold,
            embedding_client=embedding_client,
            model=embeddings_model
        )
        self.info_cov_score = self.info_cov_score_client.score
        score = (self.weights["key_point_cov_score"] * self.key_point_cov_score) + (self.weights["info_cov_score"] * self.info_cov_score)
        self.set_score(score)
        
        for answer in answers_with_scores:
            answer.set_info_cap_score(self.weights["key_point_cov_score"] * answer.kp_cov_cs_score + self.weights["info_cov_score"] * answer.in_cov_cs_score)
        
        return answers_with_scores

    def get_score(self):
        return {
            "key_point_cov_score": self.key_point_cov_score,
            "info_cov_score": self.info_cov_score,
            "info_cap_score": self.score
        }