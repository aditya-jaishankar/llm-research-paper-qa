import os
from pathlib import Path

import openai
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader

from api_keys import OPENAI_TOKEN, PINECONE_TOKEN, PINECONE_ENV

PINECONE_API_KEY = PINECONE_TOKEN
PINECONE_API_ENV = PINECONE_ENV
os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN 
DIRNAME = Path(__file__).resolve().parents[1]

# More on Document Loaders:
# https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pymupdf
if __name__ == "__main__":
    loader = DirectoryLoader(
        path=f'{DIRNAME}/data/papers/',
        glob='**/*.pdf',
        loader_cls=PyMuPDFLoader
    )

    papers = loader.load()
    print("wait")