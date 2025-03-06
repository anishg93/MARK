from abc import ABC, abstractmethod

class Dataset(ABC):
    score: float = 0.0
    
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.processed_data = None
    
    @abstractmethod
    def read_data(self, limit: int = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_data(self, **kwargs):
        raise NotImplementedError

    def get_data(self) -> list[dict]:
        if self.processed_data is None:
            self.process_data()
        return self.processed_data
