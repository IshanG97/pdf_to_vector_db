from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

app = FastAPI()

class QdrantServices:
    def __init__(self, path: str = "data/"):
        """Initialize singleton Qdrant client."""
        self.client = QdrantClient(path=path)
    
    def manage_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine", 
                         recreate: bool = False, custom_config: Optional[Dict] = None) -> None:
        if recreate and self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            
        if not self.client.collection_exists(collection_name):
            config = models.VectorParams(
                size=vector_size,
                distance=distance
            )
            if custom_config:
                config = models.VectorParams(
                    size=vector_size,
                    distance=distance,
                    **custom_config.get('vectors_config', {})
                )
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=config,
                **{k: v for k, v in custom_config.items() if k != 'vectors_config'} if custom_config else {}
            )
    
    def upsert_points(self, collection_name: str, points: List[Dict], batch_size: int = 64) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=p['id'],
                        vector=p['vector'],
                        payload=p.get('payload', {})
                    ) for p in batch
                ]
            )
    
    def search_points(self, collection_name: str, query_vector: List[float], 
                     limit: int = 10, filter: Optional[Dict] = None) -> List[Dict]:
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter
        )
        
        return [
            {
                'id': point.id,
                'score': point.score,
                'payload': point.payload
            }
            for point in search_result
        ]

# Initialize global Qdrant service
qdrant_services = QdrantServices()

# API Endpoints
@app.post("/collection/{collection_name}")
async def create_collection(collection_name: str, vector_size: int, 
                          distance: str = "Cosine", recreate: bool = False, 
                          custom_config: Optional[Dict] = None):
    try:
        qdrant_services.manage_collection(collection_name, vector_size, distance, recreate, custom_config)
        return {"message": f"Collection {collection_name} managed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/points/{collection_name}")
async def upsert_points(collection_name: str, points: List[Dict]):
    try:
        qdrant_services.upsert_points(collection_name, points)
        return {"message": f"Points upserted to {collection_name} successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/{collection_name}")
async def search(collection_name: str, query_vector: List[float], 
                limit: int = 10, filter: Optional[Dict] = None):
    try:
        results = qdrant_services.search_points(collection_name, query_vector, limit, filter)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Qdrant API server on {config['QDRANT_HOST']}:{config['QDRANT_PORT']}")
    uvicorn.run(
        app, 
        host=config["QDRANT_HOST"].replace("http://",""), 
        port=config["QDRANT_PORT"],
        log_level="info"
    )