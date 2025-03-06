import json
import pandas as pd

from .base import Dataset

class MedMCQADataSet(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.drop_data_with_null_exp = True
        super().__init__("MedMCQADataSet")

    def read_data(self, limit: int = None):
        data = []
        with open(self.file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        if limit:
            data = data[:limit]
        self.data = pd.DataFrame(data)

    def process_data(self, **kwargs):
        """
        Process the data to create the required columns
        """
        if self.data is None:
            if kwargs.get("limit") and not self.drop_data_with_null_exp:
                self.read_data(limit=kwargs["limit"])
            else:
                self.read_data()
        if self.drop_data_with_null_exp:
            self.data = self.data[self.data["exp"].notnull()]
            if kwargs.get("limit"):
                self.data = self.data.head(kwargs["limit"])
        self.data["options"] = self.data.apply(lambda x: "\n".join([f"{chr(97 + i)}) {x[f'op{chr(97 + i)}']}" for i in range(4)]), axis=1)
        intermediate_data = self.data[["question", "options", "exp"]].copy()
        intermediate_data.loc[:, "expected_answer"] = intermediate_data["exp"]
        intermediate_data.loc[:, "question"] = intermediate_data["question"] + "\n" + intermediate_data["options"]
        self.processed_data = intermediate_data[["question", "expected_answer"]].to_dict(orient="records")
