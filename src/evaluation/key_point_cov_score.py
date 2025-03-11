import json

from openai import AzureOpenAI

from .base import EvaluationMetric
from src.data.model import EvaluationData

PROMPT_KEY_POINTS_EXTRACTION = f"""
You are an AI assistant to help with extracting relevant, coherent and key topics from a given context.
You will be given a sentence with texts and your task is to extract the key topics from the given context by understanding the importance and respond in JSON format.
You have to extract top NUM_TOPICS key topics from the given context.
Make sure each topic does not exceed MAX_WORDS_PER_TOPIC words.
Response format: {{"key_points": ["topic1", "topic2", "topic3"]}}
"""

class KeyPointCoverageScore(EvaluationMetric):
    def __init__(self):
        self.num_topics = 3
        self.max_words_per_topic = 2
        self.use_cosine_similarity = False
        super().__init__("KeyPointCoverageScore")
    
    def evaluate(self, **kwargs) -> list[EvaluationData]:
        average_score = 0.0
        answers_with_scores = []
        answers : list[EvaluationData] = kwargs.get("answers")
        openai_client = kwargs.get("openai_client")
        embedding_client = kwargs.get("embedding_client")
        openai_model = kwargs.get("openai_model")
        embeddings_model = kwargs.get("embedding_model")
        cosine_similarity_threshold = kwargs.get("cosine_similarity_threshold")
        
        if not answers or not openai_client or not embedding_client or not openai_model or not embeddings_model:
            raise ValueError("Answers, OpenAI client, embedding client, OpenAI model, and embeddings model are required.")
        
        for answer in answers:
            actual = answer.generated_answer
            expected = answer.expected_answer
            if actual and expected:
                actual_key_points = self._extract_key_points(actual, openai_client, openai_model)
                expected_key_points = self._extract_key_points(expected, openai_client, openai_model)
                high_similarity_count = self._calculate_similarity(actual_key_points, expected_key_points,
                                                                    embedding_client, embeddings_model, cosine_similarity_threshold)
                if len(actual_key_points) > 0:
                    key_point_score = high_similarity_count / len(actual_key_points)
                else:
                    key_point_score = 0
            else:
                key_point_score = 0.0
            answer.set_kp_cov_cs_score(key_point_score)
            answers_with_scores.append(answer)
        
        if answers:
            average_score = sum([answer.kp_cov_cs_score for answer in answers]) / len(answers)
        self.set_score(average_score)
        print(f"Key Point Coverage Score: {average_score} from {len(answers)} answers.")
        return answers_with_scores
    
    def _convert_string_to_vector(self, string: str, embedding_client: AzureOpenAI, model: str) -> list[float]:
        return embedding_client.embeddings.create(input=string, model=model).data[0].embedding
        
    def _extract_key_points(self, text: str, openai_client, model: str) -> list[str]:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": PROMPT_KEY_POINTS_EXTRACTION.replace("NUM_TOPICS", str(self.num_topics)).replace("MAX_WORDS_PER_TOPIC", str(self.max_words_per_topic))},
                      {"role": "user", "content": f"Extract important and relevant key topics from the given text : {text}"}],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        try:
            key_points = json.loads(response)["key_points"]
        except json.JSONDecodeError:
            key_points = []
        print(f"Extracted Key Points: {key_points}")
        return key_points

    def _calculate_similarity(self, actual_key_points: list[str], expected_key_points: list[str],
                              embedding_client: AzureOpenAI, model: str, threshold: float) -> int:
        high_similarity_count = 0
        for actual_key_point in actual_key_points:
            actual_vector = self._convert_string_to_vector(actual_key_point, embedding_client, model)
            for expected_key_point in expected_key_points:
                expected_vector = self._convert_string_to_vector(expected_key_point, embedding_client, model)
                if self.use_cosine_similarity:
                    similarity = self._calculate_cosine_similarity(actual_vector, expected_vector)
                    if similarity > threshold:
                        high_similarity_count += 1
                        break
                else:
                    if actual_key_point.lower() == expected_key_point.lower():
                        high_similarity_count += 1
                        break
        return high_similarity_count
    
    def _calculate_cosine_similarity(self, actual: list[float], expected: list[float]) -> float:
        dot_product = sum([a * b for a, b in zip(actual, expected)])
        magnitude_actual = sum([a ** 2 for a in actual]) ** 0.5
        magnitude_expected = sum([b ** 2 for b in expected]) ** 0.5
        similarity = dot_product / (magnitude_actual * magnitude_expected)
        return similarity
