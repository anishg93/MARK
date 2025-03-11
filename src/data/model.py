from pydantic import BaseModel

class EvaluationData(BaseModel):
    question: str
    generated_answer: str = ""
    expected_answer: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    prompt_token_count: int = 0
    completion_token_count: int = 0
    in_cov_cs_score: float = 0.0
    kp_cov_cs_score: float = 0.0
    info_cap_score: float = 0.0
    
    def __init__(self, question: str, generated_answer: str = "", expected_answer: str | None = None,
                 in_cov_cs_score: float = 0.0, kp_cov_cs_score: float = 0.0, info_cap_score: float = 0.0,
                 session_id: str | None = None, turn_id: str | None = None,
                 prompt_token_count: int = 0, completion_token_count: int = 0):
        super().__init__(question=question, expected_answer=expected_answer,
                         generated_answer=generated_answer, in_cov_cs_score=in_cov_cs_score,
                         kp_cov_cs_score=kp_cov_cs_score, info_cap_score=info_cap_score,
                         session_id=session_id, turn_id=turn_id,
                         prompt_token_count=prompt_token_count, completion_token_count=completion_token_count)
    
    def set_generated_answer(self, answer: str):
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
            "generated_answer": self.generated_answer,
            "expected_answer": self.expected_answer,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "prompt_token_count": self.prompt_token_count,
            "completion_token_count": self.completion_token_count,
            "in_cov_cs_score": self.in_cov_cs_score,
            "kp_cov_cs_score": self.kp_cov_cs_score,
            "info_cap_score": self.info_cap_score
        }
