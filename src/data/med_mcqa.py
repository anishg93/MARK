import json
import pandas as pd
import tiktoken

from .base import Dataset

class MedMCQADataSet(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.clean_data = True
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model("gpt-3.5"))
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
            if kwargs.get("limit") and not self.clean_data:
                self.read_data(limit=kwargs["limit"])
            else:
                self.read_data()
        if self.clean_data:
            # Remove rows where none of the options are correct
            self.data = self.data[self.data["exp"].notnull()]
            # Remove rows where the question contains reasoning keyword "what"
            self.data = self.data[~self.data["question"].str.contains("what ", case=False)]
            # Remove rows where the question token count is greater than 20
            self.data = self.data[self.data["question"].apply(lambda x: len(self.tokenizer.encode(x)) <= 10)]
            if kwargs.get("limit"):
                self.data = self.data.head(kwargs["limit"])
        self.data["options"] = self.data.apply(lambda x: "\n".join([f"{chr(97 + i)}) {x[f'op{chr(97 + i)}']}" for i in range(4)]), axis=1)
        intermediate_data = self.data[["question", "options", "exp"]].copy()
        intermediate_data.loc[:, "expected_answer"] = intermediate_data["exp"]
        intermediate_data.loc[:, "question"] = intermediate_data["question"] + "\n" + intermediate_data["options"]
        self.processed_data = intermediate_data[["question", "expected_answer"]].to_dict(orient="records")
