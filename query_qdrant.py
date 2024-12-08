from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import torch

# Check for MPS device (Apple Silicon GPU support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the embedding model and move it to the appropriate device
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)  # Move model to MPS or CPU based on availability

# Initialize Qdrant client
client = QdrantClient(path="qdrant_storage")  # Use the persistent storage directory
collection_name = "pdf_embeddings"  # Updated to match the collection name in process_pdfs.py

# Function to Query Qdrant
def query_qdrant(query_text, limit=5):
    """
    Query the Qdrant database for similar sentences with metadata.

    Args:
        query_text (str): The text query to search for.
        limit (int): The number of top results to retrieve.

    Returns:
        list: Search results with filenames, page numbers, sentences, and additional metadata.
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
        {
            "filename": result.payload["filename"],
            "page_number": result.payload["page_number"],
            "sentence": result.payload["sentence"],
            "title": result.payload.get("title", ""),
            "author": result.payload.get("author", ""),
            "creation_date": result.payload.get("creation_date", ""),
            "modification_date": result.payload.get("modification_date", ""),
            "keywords": result.payload.get("keywords", ""),
            "dimensions": result.payload.get("dimensions", {}),
            "score": result.score
        }
        for result in results
    ]
    return formatted_results


# Example Query
if __name__ == "__main__":
    query_text = "What did Teri say the cash used in operations in the third quarter of Q3 was?"  # Replace with your query
    search_results = query_qdrant(query_text)

    # Display Search Results
    print("Search Results:")
    for result in search_results:
        print(
            f"File: {result['filename']}, "
            f"Page: {result['page_number']}, "
            f"Sentence: {result['sentence']}, "
            f"Title: {result['title']}, "
            f"Author: {result['author']}, "
            f"Creation Date: {result['creation_date']}, "
            f"Modification Date: {result['modification_date']}, "
            f"Keywords: {result['keywords']}, "
            f"Dimensions: {result['dimensions']}, "
            f"Score: {result['score']}"
        )
