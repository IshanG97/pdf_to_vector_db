What this does: 
- Parses PDFs into images and then into image embeddings in json format. Also has basic QC checks

Quick setup:
1. Create a pipenv shell
2. Run pip install -r requirements.txt

Converting PDFs to PNG images:
1. Make sure you have your pdf files inside the input_pdfs folder
2. Run pdf_to_image.py
3. You should see the images in the output_images folder
3. Note: we don't need to patch the images as ColPali does this for us
4. Note: You need to install poppler for pdf_to_image.py
    - macOS: brew install poppler
    - Linux: sudo apt-get install poppler-utils
    - Windows: look at https://poppler.freedesktop.org/

Creating embeddings:
1. Run create_embeddings.py - note to change the accelerator depending on your device e.g. "CUDA:0" if you have an Nvidia GPU or "mps" for Apple Silicon macOS devices

Run a query:
1. Run query.py - note to change the query. You should see the top k results output in the terminal

To do:
- batch processing in create_embeddings.py

Extras:
- if your GPU has enough memory to simultaneously create the embeddings AND process the query, create_embeddings_and_query.py does the job