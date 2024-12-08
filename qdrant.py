from qdrant_client import QdrantClient
import json

# Connect to Qdrant (default local instance on port 6333)
client = QdrantClient(url="http://localhost:6333", timeout=600)  # Increase timeout to 60 seconds


# Load embeddings from a JSON file
with open("image_embeddings.json", "r") as f:
    embeddings = json.load(f)

# Flatten the embeddings
for embedding in embeddings:
    embedding["embedding"] = [value for sublist in embedding["embedding"] for value in sublist]

# Compute the vector size
vector_size = len(embeddings[0]["embedding"])
print(f"Vector size after flattening: {vector_size}")

# Define collection name
collection_name = "image_embeddings"

# Delete the existing collection if it exists
if client.collection_exists(collection_name):
    client.delete_collection(collection_name=collection_name)
    print(f"Deleted existing collection: {collection_name}")

# Create the collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config={"size": vector_size, "distance": "Cosine"}  # Use cosine similarity
)

# Prepare vectors and payloads for upload
points = [
    {
        "id": idx,  # Unique ID for each vector
        "vector": item["embedding"],  # Embedding vector
        "payload": {  # Optional metadata
            "path": item["path"],
            **item.get("metadata", {})
        }
    }
    for idx, item in enumerate(embeddings)
]

# Upload embeddings to Qdrant
client.upsert(
    collection_name=collection_name,
    points=points
)

# Example query embedding
query_embedding = [-0.0191, -0.0214, 0.0791, ...]

# Perform a search for the top-5 most similar vectors
results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5
)

# Print results
for result in results:
    print(f"ID: {result.id}, Score: {result.score}, Metadata: {result.payload}")

info = client.get_collection(collection_name=collection_name)
print(info)
