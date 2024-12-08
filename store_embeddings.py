import os
import torch
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from PIL import Image
from uuid import uuid4
import stamina
from colpali_engine.models import ColPali, ColPaliProcessor

# Initialize Qdrant
client = QdrantClient(path="qdrant_storage")
collection_name = "colpali_embeddings"
dim = 128  # Dimensionality of ColPali embeddings

# Initialize ColPali Model and Processor
model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",#cuda:0
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

client.create_collection(
    collection_name=collection_name,
    on_disk_payload=True,  # store the payload on disk
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        on_disk=True, # move original vectors to disk
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True  # keep only quantized vectors in RAM
            ),
        ),
    ),
)

@stamina.retry(on=Exception, attempts=3) # retry mechanism if an exception occurs during the operation
def upsert_to_qdrant(batch):
    try:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False
    return True

# Prepare Documents (Images of Pages)
image_dir = "output_images"
image_files = os.listdir(image_dir)
images = [{"image": Image.open(os.path.join(image_dir, name)), "filename": name} for name in image_files]

batch_size = 4  # Adjust based on your GPU memory constraints

# Use tqdm to create a progress bar
with tqdm(total=len(images), desc="Indexing Progress") as pbar:
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        batch_images = [item["image"] for item in batch]
        batch_filenames = [item["filename"] for item in batch]

        # Process and encode images
        with torch.no_grad():
            processed_images = processor.process_images(batch_images).to(model.device)
            image_embeddings = model(**processed_images)

        # Prepare points for Qdrant
        points = []
        for j, (embedding, filename) in enumerate(zip(image_embeddings, batch_filenames)):
            # Convert the embedding to a list of vectors
            multivector = embedding.cpu().float().numpy().tolist()
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=multivector,  # List of vectors
                    payload={
                        "filepath": os.path.join(image_dir, filename),
                    },  # Metadata
                )
            )

        # Upload points to Qdrant
        try:
            upsert_to_qdrant(points)
        except Exception as e:
            print(f"Error during upsert: {e}")
            continue

        # Update the progress bar
        pbar.update(batch_size)

print("Indexing complete!")

client.update_collection(
    collection_name=collection_name,
    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
)