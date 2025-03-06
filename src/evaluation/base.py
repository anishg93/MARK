from abc import ABC, abstractmethod

class EvaluationMetric(ABC):
    score: float = 0.0
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError

    def set_score(self, score: float):
        self.score = score

    def __str__(self):
        return self.name
