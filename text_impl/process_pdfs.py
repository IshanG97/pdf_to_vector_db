import os
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
import torch
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')  # Download the missing tokenizer data

# Check for MPS device (Apple Silicon GPU support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

# Initialize Qdrant client
client = QdrantClient(path="../qdrant_storage")  # Persistent storage
collection_name = "pdf_embeddings"

# Check if collection exists, delete and create it
if client.collection_exists(collection_name):
    client.delete_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config={"size": model.get_sentence_embedding_dimension(), "distance": "Cosine"}
)

# Path to input folder
input_folder = "../input_pdfs"

# Helper function to extract metadata
def extract_metadata(file_path):
    # Extract basic metadata using PyMuPDF
    doc = fitz.open(file_path)
    pdf_reader = PdfReader(file_path)

    metadata = {
        "filename": os.path.basename(file_path),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "keywords": doc.metadata.get("keywords", ""),
        "creator": doc.metadata.get("creator", ""),
        "producer": doc.metadata.get("producer", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "modification_date": doc.metadata.get("modDate", ""),
        "page_count": len(doc),
        "custom_metadata": pdf_reader.metadata,  # Extract custom metadata via PyPDF2
        "pages": [],
    }

    # Process each page for additional metadata
    for page_num, page in enumerate(doc, start=1):
        page_data = {
            "page_number": page_num,
            "text": page.get_text(),
            "dimensions": {"width": page.rect.width, "height": page.rect.height},
            "images": [],
            "sentences": [],
        }

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_info = {
                "image_index": img_index,
                "image_bytes": base_image["image"],
                "width": base_image.get("width"),
                "height": base_image.get("height"),
            }
            page_data["images"].append(image_info)

        # Extract sentences
        sentences = sent_tokenize(page_data["text"])
        page_data["sentences"] = sentences
        metadata["pages"].append(page_data)

    return metadata


# Process all PDFs in the folder
for pdf_file in os.listdir(input_folder):
    if pdf_file.endswith(".pdf"):
        file_path = os.path.join(input_folder, pdf_file)

        # Extract metadata
        metadata = extract_metadata(file_path)

        # Store page-level embeddings and metadata in Qdrant
        for page in metadata["pages"]:
            sentences = page["sentences"]
            embeddings = model.encode(sentences, device=device, batch_size=16)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=list(embedding),
                    payload={
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
                )
                for embedding, sentence in zip(embeddings, sentences)
            ]

            client.upsert(
                collection_name=collection_name,
                points=points
            )
        print(f"Processed: {pdf_file}")
