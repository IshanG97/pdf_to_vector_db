-- Built during EF Hackathon London 2024 --

Ingestion engine for rich PDFs. It has two implementations:
1. Parse PDFs into images using Pillow and use ColPali (Vision Language Model) to process the images into vector embeddings - captures the context of text AND visual elements
2. SentencesTransformers library, PyMuPDF and PyPDF2 to convert the text into vector embeddings - captures the context of just the text, but has more metadata

The vector embeddings created above can be passed into a Qdrant vector database. 

During the hackathon this was integrated into an agentic system, link here: https://github.com/KenjiPcx/ef-fall-hack

Pre-requisite steps:
1. Create a `pipenv shell`
2. Run `pip install -r requirements.txt`
3. Store your pdf files inside an `input_pdfs` folder at the root of the project
4. Get Docker running with Qdrant
    - `docker pull qdrant/qdrant:latest`
    - `docker run -p 6333:6333 -d qdrant/qdrant`

Method #1 of ingestion - using ColPali:
1. Run `vlm_impl/pdf_to_image.py` to parse the PDFs into images to feed into the ingestion engine
2. Run `vlm_impl/store_embeddings.py` - note to change the torch accelerator depending on your device e.g. "mps" for Apple Silicon macOS devices

Method #2 of ingestion - using SentencesTransformer:
1. Run `text_impl/process_pdfs.py` - note to change the torch accelerator depending on your device e.g. "mps" for Apple Silicon macOS devices

Validation:
1. Run the cells `query_qdrant.ipynb` - note to update the query with your specific request
