import torch
from torch.utils.data import DataLoader
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset


# Initialize Qdrant Client
client = QdrantClient(path="qdrant_storage")
collection_name = "colpali_embeddings"

# Define Retriever
class QdrantColbertRetriever:
    def __init__(self, qdrant_client, collection_name):
        self.client = qdrant_client
        self.collection_name = collection_name

    def search(self, query_vector, topk):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=topk,
        )
        return [(res.score, res.payload["doc_id"], res.payload["filepath"]) for res in results]


# Initialize ColPali Model and Processor
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Prepare Queries
queries = [
    "What was Gavin referring to when starting another medication?"
]

query_dataloader = DataLoader(
    dataset=ListDataset[str](queries),
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor.process_queries(x),
)

query_embeddings = []
for batch_query in query_dataloader:
    with torch.no_grad():
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
    query_embeddings.extend(list(torch.unbind(embeddings_query.to("mps" if torch.backends.mps.is_available() else "cpu"))))

# Initialize Retriever
retriever = QdrantColbertRetriever(collection_name=collection_name, qdrant_client=client)

# Perform Search for Each Query
for query in query_embeddings:
    query = query.float().numpy()
    results = retriever.search(query, topk=1)

    # Display Results
    print("Search Results:")
    for score, doc_id, filepath in results:
        print(f"Filepath: {filepath}, Score: {score}")
