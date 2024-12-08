What this does: 
- Parses PDFs into vector embeddings using PyMuPDF, PyPDF2 and the SentencesTransformers library
- Load these embeddings into a vector database - QDRANT

How to create the vector database:
1. (optional) Create a pipenv shell
2. (optional) Run pip install -r requirements.txt
3. Store your pdf files inside an "input_pdfs" folder at the root of the project
4. Get Docker running with QDRANT
    - docker pull qdrant/qdrant:latest
    - docker run -p 6333:6333 -d qdrant/qdrant
4. Run process_pdfs.py - note to change the torch accelerator depending on your device e.g. "mps" for Apple Silicon macOS devices

Run a query:
1. Run query_qdrant.py - note to change the query. You should see the top results in the terminal