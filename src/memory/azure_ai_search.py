from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex,
)
from azure.search.documents.models import VectorizedQuery
from .model import Memory, ResidualMemory, UserQuestionMemory, AssistantResponseMemory


class AzureAISearch:
    def __init__(self, endpoint: str, key: str, index_name: str):
        self.index_name = index_name
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    def create_index(self, model: str):
        try:
            index_definition = self.index_client.get_index(self.index_name)
        except Exception as e:
            index_definition = self._create_index_definition(self.index_name, model)
            self.index_client.create_index(index_definition)
    
    async def upload_memories(self, memories: list[Memory]):
        if not memories:
            print("******** No memories to upload **********")
            return
        self.search_client.upload_documents(documents=[mem.to_dict() for mem in memories])
    
    def search_memory(self, query: str, query_vector: list[float], type: str,
                      user: str = None, agent: str = None, top: int = 3, relevance_threshold: int = 0.7) -> list[Memory]:
        search_vector = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top, fields="memoryVector")
        query_filter = f"type eq '{type}'"
        if user:
            query_filter += f" and user eq '{user}'"
        if agent:
            query_filter += f" and agent eq '{agent}'"
        result = self.search_client.search(
            search_text=query,
            vector_queries=[search_vector],
            filter=query_filter,
            top=top,
            select=["memory", "classification"])
        memories = []
        for result in result:
            if result["@search.score"] < relevance_threshold:
                continue
            if type == ResidualMemory.type:
                memory = ResidualMemory(
                    memory=result["memory"],
                    classification=result["classification"]
                )
            elif type == UserQuestionMemory.type:
                memory = UserQuestionMemory(
                    memory=result["memory"],
                    classification=result["classification"]
                )
            elif type == AssistantResponseMemory.type:
                memory = AssistantResponseMemory(
                    memory=result["memory"],
                    classification=result["classification"]
                )
            else:
                raise ValueError(f"Invalid memory type: {type}")
            memories.append(memory)
        return memories

    def _create_index_definition(self, index_name: str, model: str) -> SearchIndex:
        dimensions = 1536
        if model == "text-embedding-3-large":
            dimensions = 3072

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="type", type=SearchFieldDataType.String),
            SimpleField(name="user", type=SearchFieldDataType.String),
            SimpleField(name="agent", type=SearchFieldDataType.String),
            SimpleField(name="classification", type=SearchFieldDataType.String),
            SearchableField(name="memory", type=SearchFieldDataType.String),
            SearchField(
                name="memoryVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
        ]
        
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=1000,
                        ef_search=1000,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="myExhaustiveKnn",
                    kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                    parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="myExhaustiveKnn",
                ),
            ],
        )

        return SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )
