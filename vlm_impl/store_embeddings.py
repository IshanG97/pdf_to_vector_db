import os
import torch
from PIL import Image
from uuid import uuid4
from tqdm import tqdm
import stamina
import requests
from colpali_engine.models import ColPali, ColPaliProcessor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

QDRANT_API_URL = f"{config['QDRANT_HOST']}:{config['QDRANT_PORT']}"

def init_colpali(model_name="vidore/colpali-v1.2", device="mps"):
    """Initialize ColPali model and processor."""
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor

def init_collection(collection_name="colpali_embeddings"):
    """Initialize Qdrant collection through API."""
    custom_config = {
        "on_disk_payload": True,
        "vectors_config": {
            "on_disk": True,
            "multivector_config": {
                "comparator": "MAX_SIM"
            },
            "quantization_config": {
                "binary": {
                    "always_ram": True
                }
            }
        }
    }
    
    response = requests.post(
        f"{QDRANT_API_URL}/collection/{collection_name}",
        json={
            "vector_size": 128,
            "distance": "Cosine",
            "recreate": True,
            "custom_config": custom_config
        }
    )
    response.raise_for_status()

def load_images(image_dir):
    """Load all images from directory."""
    image_files = os.listdir(image_dir)
    return [
        {"image": Image.open(os.path.join(image_dir, name)), "filename": name} 
        for name in image_files
    ]

@stamina.retry(on=Exception, attempts=3)
def process_batch(batch_images, batch_filenames, model, processor, image_dir):
    """Process a batch of images and create points for Qdrant."""
    points = []
    with torch.no_grad():
        processed_images = processor.process_images(batch_images).to(model.device)
        image_embeddings = model(**processed_images)
        
        for embedding, filename in zip(image_embeddings, batch_filenames):
            multivector = embedding.cpu().float().numpy().tolist()
            points.append({
                'id': str(uuid4()),
                'vector': multivector,
                'payload': {
                    'filepath': os.path.join(image_dir, filename),
                }
            })
    return points

def upload_points(points, collection_name):
    """Upload points through API."""
    response = requests.post(
        f"{QDRANT_API_URL}/points/{collection_name}",
        json=points
    )
    response.raise_for_status()

def process_images(images, image_dir, model, processor, collection_name, batch_size=4):
    """Process all images in batches and index them through API."""
    with tqdm(total=len(images), desc="Indexing Progress") as pbar:
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_images = [item["image"] for item in batch]
            batch_filenames = [item["filename"] for item in batch]
            
            try:
                points = process_batch(batch_images, batch_filenames, model, processor, image_dir)
                upload_points(points, collection_name)
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
                
            pbar.update(len(batch))

if __name__ == "__main__":
        # Initialize model
    model, processor = init_colpali()
    collection_name = "colpali_embeddings"
    
    # Initialize collection
    init_collection(collection_name)
    
    # Load and process images
    image_dir = "../output_images"
    images = load_images(image_dir)
    
    # Process images
    process_images(images, image_dir, model, processor, collection_name)
    print("Indexing complete!")