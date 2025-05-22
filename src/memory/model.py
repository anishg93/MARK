from typing import List
from uuid import uuid4
from pydantic import BaseModel
from datetime import datetime

class Memory(BaseModel):
    id: str
    type: str
    user: str
    agent: str
    recall: int = 0
    classification: str
    created_at: str = None
    memory: str
    memoryVector: List[float]
    search_score: float = 0.0
    
    def __init__(self, type: str, memory: str, user: str = "", memoryVector: List[float] = [],
                 classification: str = "", agent: str = "", recall: int = 0, created_at: str = None, search_score: float = 0.0):
        if created_at is None:
            created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        super().__init__(id=str(uuid4()), type=type, user=user,
                         classification=classification, memory=memory, memoryVector=memoryVector,
                         agent=agent, recall=recall, created_at=created_at, search_score=search_score)
    
    def set_classification(self, classification: str):
        self.classification = classification
    
    def set_type(self, type: str):
        self.type = type
    
    def set_user(self, user: str):
        self.user = user
    
    def set_agent(self, agent: str):
        self.agent = agent
    
    def set_recall(self, recall: int):
        self.recall = recall
    
    def set_search_score(self, search_score: float):
        self.search_score = search_score
    
    def set_memory_vector(self, memoryVector: List[float]):
        self.memoryVector = memoryVector
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "user": self.user,
            "agent": self.agent,
            "recall": self.recall,
            "classification": self.classification,
            "created_at": self.created_at,
            "memory": self.memory,
            "memoryVector": self.memoryVector
        }

class ResidualMemory(Memory):
    type: str = "residual"
    def __init__(self, memory: str, user: str = "", memoryVector: List[float] = [],
                 classification: str = "", agent: str = "", recall: int = 0,
                 created_at: str = None, search_score: float = 0.0):
        super().__init__(type="residual", memory=memory, user=user, memoryVector=memoryVector,
                         classification=classification, agent=agent, recall=recall,
                         created_at=created_at, search_score=search_score)

class UserQuestionMemory(Memory):
    type: str = "user_question"
    def __init__(self, memory: str, user: str = "", memoryVector: List[float] = [],
                 classification: str = "", agent: str = "", recall: int = 0,
                 created_at: str = None, search_score: float = 0.0):
        super().__init__(type="user_question", memory=memory, user=user, memoryVector=memoryVector,
                         classification=classification, agent=agent, recall=recall,
                         created_at=created_at, search_score=search_score)

class AssistantResponseMemory(Memory):
    type: str = "assistant_response"
    def __init__(self, memory: str, user: str = "", memoryVector: List[float] = [],
                 classification: str = "", agent: str = "", recall: int = 0,
                 created_at: str = None, search_score: float = 0.0):
        super().__init__(type="assistant_response", memory=memory, user=user, memoryVector=memoryVector,
                         classification=classification, agent=agent, recall=recall,
                         created_at=created_at, search_score=search_score)
