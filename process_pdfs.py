import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
import torch

# Check for MPS device (Apple Silicon GPU support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the embedding model and move it to the appropriate device
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)  # Move model to MPS or CPU based on availability

# Initialize Qdrant client
client = QdrantClient(path="qdrant_storage")  # Use a persistent storage directory
collection_name = "pdf_embeddings"

# Check if collection exists, delete and create it
if client.collection_exists(collection_name):
    client.delete_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config={"size": model.get_sentence_embedding_dimension(), "distance": "Cosine"}
)

# Path to input folder
input_folder = "input_pdfs"

# Process all PDFs in the folder
for pdf_file in os.listdir(input_folder):
    if pdf_file.endswith(".pdf"):
        file_path = os.path.join(input_folder, pdf_file)
        doc = fitz.open(file_path)

        # Extract text and metadata
        text = ""
        metadata = doc.metadata
        for page in doc:
            text += page.get_text()

        # Generate embeddings on the appropriate device
        embedding = model.encode(
            text,
            device=device,  # Ensure encoding uses MPS or CPU
            batch_size=16   # Optimize batch size for performance
        )

        # Generate a valid UUID from the filename
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, pdf_file))

        # Prepare point using PointStruct
        point = PointStruct(
            id=point_id,  # Use the UUID as the ID
            vector=list(embedding),  # Convert to list if needed
            payload={
                "filename": pdf_file,
                "metadata": metadata,
            }
        )

        # Upload to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[point]  # Provide as a list of PointStruct objects
        )
        print(f"Processed: {pdf_file}")
