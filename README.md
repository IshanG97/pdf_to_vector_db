-- EF Hackathon 2024 --

This repo is an ingestion engine for rich PDFs. It has the following features:
- Parse PDFs into images using Pillow
- Use either:
    1. ColPali (Vision Language Model) to process the images into vector embeddings
    2. SentencesTransformers library, PyMuPDF and PyPDF2 to convert just the text into vector embeddings
- Pass the vector embeddings into a Qdrant vector database that will be used by agentic system

Pre-requisite steps:
1. (optional) Create a pipenv shell
2. (optional) Run pip install -r requirements.txt
3. Store your pdf files inside an "input_pdfs" folder at the root of the project
4. Run "pdf_to_image.py" to create your images to feed into the ingestion engine

Method #1 of ingestion - using ColPali:
1. Run store_embeddings.py - note to change the torch accelerator depending on your device e.g. "mps" for Apple Silicon macOS devices

Method #2 of ingestion - using SentencesTransformer:
1. Run process_pdfs.py - note to change the torch accelerator depending on your device e.g. "mps" for Apple Silicon macOS devices

Validation:
1. Run query_qdrant.py - note to update the query with your specific request

How to create the vector database:
1. Get Docker running with Qdrant
    - docker pull qdrant/qdrant:latest
    - docker run -p 6333:6333 -d qdrant/qdrant