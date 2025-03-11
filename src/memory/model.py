from typing import List
from uuid import uuid4
from pydantic import BaseModel

class Memory(BaseModel):
    id: str
    type: str
    user: str
    classification: str
    memory: str
    memoryVector: List[float]
    
    def __init__(self, type: str, memory: str, user: str = "", memoryVector: List[float] = [], classification: str = ""):
        super().__init__(id=str(uuid4()), type=type, user=user,
                         classification=classification, memory=memory, memoryVector=memoryVector)
    
    def set_classification(self, classification: str):
        self.classification = classification
    
    def set_type(self, type: str):
        self.type = type
    
    def set_memory_vector(self, memoryVector: List[float]):
        self.memoryVector = memoryVector
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "user": self.user,
            "classification": self.classification,
            "memory": self.memory,
            "memoryVector": self.memoryVector
        }

class ResidualMemory(Memory):
    type: str = "residual"
    def __init__(self, memory: str, user: str = "", memoryVector: List[float] = [], classification: str = ""):
        super().__init__(type="residual", memory=memory, user=user, memoryVector=memoryVector, classification=classification)

class UserQuestionMemory(Memory):
    type: str = "user_question"
    def __init__(self, memory: str, user: str = "", memoryVector: List[float] = [], classification: str = ""):
        super().__init__(type="user_question", memory=memory, user=user, memoryVector=memoryVector, classification=classification)

class AssistantResponseMemory(Memory):
    type: str = "assistant_response"
    def __init__(self, memory: str, user: str = "", memoryVector: List[float] = [], classification: str = ""):
        super().__init__(type="assistant_response", memory=memory, user=user, memoryVector=memoryVector, classification=classification)
