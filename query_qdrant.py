from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import torch

# Check for MPS device (Apple Silicon GPU support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the embedding model and move it to the appropriate device
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)  # Move model to MPS or CPU based on availability

# Initialize Qdrant client
client = QdrantClient(path="qdrant_storage")  # Use a persistent storage directory
collection_name = "pdf_embeddings"

# Function to Query Qdrant
def query_qdrant(query_text, limit=5):
    """
    Query the Qdrant database for similar documents.
    
    Args:
        query_text (str): The text query to search for.
        limit (int): The number of top results to retrieve.

    Returns:
        list: Search results with filenames and scores.
    """
    # Generate query embedding
    query_embedding = model.encode(query_text, device=device)

    # Perform search in Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit
    )

    # Format results
    formatted_results = [
        {"filename": result.payload["filename"], "score": result.score}
        for result in results
    ]
    return formatted_results


# Example Query
if __name__ == "__main__":
    query_text = "What did Teri say the cash used in the third quarter was?"  # Replace with your query
    search_results = query_qdrant(query_text)

    # Display Search Results
    print("Search Results:")
    for result in search_results:
        print(f"File: {result['filename']}, Score: {result['score']}")
