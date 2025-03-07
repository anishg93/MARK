from pydantic import BaseModel

class EvaluationData(BaseModel):
    question: str
    generated_answer: str = ""
    expected_answer: str | None = None
    in_cov_cs_score: float = 0.0
    kp_cov_cs_score: float = 0.0
    info_cap_score: float = 0.0
    
    def __init__(self, question: str, actual_answer: str = "", expected_answer: str | None = None,
                 in_cov_cs_score: float = 0.0, kp_cov_cs_score: float = 0.0, info_cap_score: float = 0.0):
        super().__init__(question=question, expected_answer=expected_answer,
                         actual_answer=actual_answer, in_cov_cs_score=in_cov_cs_score,
                         kp_cov_cs_score=kp_cov_cs_score, info_cap_score=info_cap_score)
    
    def set_actual_answer(self, answer: str):
        self.generated_answer = answer
    
    def set_expected_answer(self, answer: str):
        self.expected_answer = answer
    
    def set_in_cov_cs_score(self, score: float):
        self.in_cov_cs_score = score
    
    def set_kp_cov_cs_score(self, score: float):
        self.kp_cov_cs_score = score
    
    def set_info_cap_score(self, score: float):
        self.info_cap_score = score

    def to_dict(self):
        return {
            "question": self.question,
            "actual_answer": self.generated_answer,
            "expected_answer": self.expected_answer,
            "in_cov_cs_score": self.in_cov_cs_score,
            "kp_cov_cs_score": self.kp_cov_cs_score,
            "info_cap_score": self.info_cap_score
        }
