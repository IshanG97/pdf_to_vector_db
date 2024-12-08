import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Check for MPS device (Apple Silicon GPU support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

# Initialize Qdrant client
client = QdrantClient(path="qdrant_storage")  # Persistent storage
collection_name = "pdf_embeddings"

# Query the database
def query_database(query, top_k=5):
    # Encode the query
    query_embedding = model.encode(query, device=device).tolist()
    
    # Perform search in Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Display results
    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"  Score: {result.score}")
        print(f"  Payload:")
        for key, value in result.payload.items():
            print(f"    {key}: {value}")
        print()

# Example query
query_text = "How much cash did Teri say was used in operations in Q2?"
query_database(query_text, top_k=3)
