# config.py
import os
from dotenv import load_dotenv
from typing import Dict

def load_config():
    # Load base .env file first
    load_dotenv(".env")
    
    # Get current environment from .env
    env = os.getenv("ENVIRONMENT", "dev")
    
    # Then load environment-specific file which can override base settings
    env_file = f".env.{env}"
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    config = {
        "ENVIRONMENT": env,
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
    }

    return config

config = load_config()