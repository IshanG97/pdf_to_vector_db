import os
from uuid import uuid4
from typing import List
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset


# Initialize Qdrant Client
client = QdrantClient(path="qdrant_storage")  # Persistent storage
collection_name = "colpali_embeddings"

# Define Retriever
class QdrantColbertRetriever:
    def __init__(self, qdrant_client, collection_name, dim=128):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.dim = dim

        # Check if the collection exists
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except ValueError:
            self.create_collection()

    def create_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def insert(self, data):
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=data["colbert_vecs"][i],
                payload={
                    "doc_id": data["doc_id"],
                    "filepath": data["filepath"],
                    "seq_id": i,
                },
            )
            for i in range(len(data["colbert_vecs"]))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, topk):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=topk,
        )
        return [(res.score, res.payload) for res in results]

# Initialize ColPali Model and Processor
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Prepare Documents (Images of Pages)
image_dir = "output_images"
images = [Image.open(os.path.join(image_dir, name)) for name in os.listdir(image_dir)]

doc_dataloader = DataLoader(
    dataset=ListDataset[str](images),
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: processor.process_images(x),
)

doc_embeddings: List[torch.Tensor] = []
for batch_doc in tqdm(doc_dataloader, desc="Processing Documents"):
    with torch.no_grad():
        batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    doc_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

# Initialize Retriever
retriever = QdrantColbertRetriever(collection_name=collection_name, qdrant_client=client)

# Insert Document Embeddings into Qdrant
filepaths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
for i, filepath in enumerate(filepaths):
    data = {
        "colbert_vecs": doc_embeddings[i].float().numpy(),
        "doc_id": i,
        "filepath": filepath,
    }
    retriever.insert(data)

print(f"Successfully inserted {len(filepaths)} documents into Qdrant.")
