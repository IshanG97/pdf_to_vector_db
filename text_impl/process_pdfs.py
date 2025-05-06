import os
from sentence_transformers import SentenceTransformer
import uuid
import torch
import requests
from typing import List, Dict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.extract_metadata import extract_metadata
from utils.config import config

QDRANT_API_URL = f"http://{config['QDRANT_HOST']}:{config['QDRANT_PORT']}"

def init_model():
    """Initialize the sentence transformer model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.to(device)
    return model, device

def process_single_pdf(
    file_path: str,
    model: SentenceTransformer,
    device: torch.device,
    collection_name: str
) -> None:
    """Process a single PDF file and store its embeddings."""
    try:
        # Extract metadata
        metadata = extract_metadata(file_path)

        # Process each page
        for page in metadata["pages"]:
            sentences = page["sentences"]
            if not sentences:  # Skip empty pages
                continue
                
            # Generate embeddings
            embeddings = model.encode(sentences, device=device, batch_size=16)

            # Create points for Qdrant
            points = [
                {
                    'id': str(uuid.uuid4()),
                    'vector': list(embedding),
                    'payload': {
                        "filename": metadata["filename"],
                        "page_number": page["page_number"],
                        "sentence": sentence,
                        "title": metadata["title"],
                        "author": metadata["author"],
                        "creation_date": metadata["creation_date"],
                        "modification_date": metadata["modification_date"],
                        "keywords": metadata["keywords"],
                        "custom_metadata": metadata["custom_metadata"],
                        "dimensions": page["dimensions"],
                    },
                }
                for embedding, sentence in zip(embeddings, sentences)
            ]

            # Upload to Qdrant via API
            response = requests.post(
                f"{QDRANT_API_URL}/points/{collection_name}",
                json=points
            )
            response.raise_for_status()
        
        print(f"Processed: {Path(file_path).name}")
    except Exception as e:
        print(f"Error processing {Path(file_path).name}: {e}")

def verify_collection(
    collection_name: str,
    vector_size: int
) -> bool:
    """Verify that the collection exists with correct configuration."""
    try:
        # Check if collection exists
        response = requests.get(f"{QDRANT_API_URL}/collection/{collection_name}/info")
        response.raise_for_status()
        print(f"Connected to collection: {collection_name}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to collection: {e}")
        print("Creating collection...")
        try:
            # Create collection if it doesn't exist
            response = requests.post(
                f"{QDRANT_API_URL}/collection/{collection_name}",
                json={
                    "vector_size": vector_size,
                    "distance": "Cosine",
                    "recreate": True
                }
            )
            response.raise_for_status()
            print("Collection created successfully")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error creating collection: {e}")
            print("Please ensure the Qdrant API service is running and accessible")
            print(f"Vector size: {vector_size}")
            print("Distance: Cosine")
            return False

def process_pdfs(
    verify_collection_flag: bool = True,
    input_folder: str = "../input_pdfs",
    collection_name: str = "pdf_embeddings"
) -> None:
    """Process PDFs and store embeddings in Qdrant."""
    # Initialize model
    model, device = init_model()
    
    # Verify collection if requested
    if verify_collection_flag:
        if not verify_collection(
            collection_name,
            model.get_sentence_embedding_dimension()
        ):
            return

    # Process each PDF in the input folder
    input_path = Path(input_folder)
    for pdf_file in input_path.glob("*.pdf"):
        process_single_pdf(
            str(pdf_file),
            model,
            device,
            collection_name
        )

if __name__ == "__main__":
    process_pdfs()