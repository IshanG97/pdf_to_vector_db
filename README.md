What this does: 
- Parses PDFs into images using Pillow and then uses ColPali to process the images into vector embeddings in json format that will subsequently be loaded into QDRANT - vector database
- Also has basic QC checks
>>>>>>> 59fca48 (Testing the waters)

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