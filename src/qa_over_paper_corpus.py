import os
from pathlib import Path

import pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

from api_keys import OPENAI_TOKEN, PINECONE_ENV, PINECONE_TOKEN
from utils import PineconeCustom

PINECONE_API_KEY = PINECONE_TOKEN
PINECONE_API_ENV = PINECONE_ENV
os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN
DIRNAME = Path(__file__).resolve().parents[1]
EMBED_PAPERS = True


def print_answer(search_index: Pinecone, question: str):
    """
    This function implements the load QA with sources method from langchain on a custom
    set of documents provided. Instead of the providing the docs themselves, the
    pre-computer search index (i.e. the vectorstore index) stored in pinecone is passed.
    This has several advantages: 1. It's faster 2. It's cheaper, since the pre-computed vector index can be loaded instead of making API calls to calculate the embeddings
    for text. The method returns the top k documents based on a similarity search and
    the answer is then wrapped up in meaningful text using langchain.

    Args:
        search_index:
            The Pinecone search_index containing the pre computed vectors and documents.
            This essentially serves as the corpus of documents that will be considered.
        question:
            The question being asked that will be answered over the corpus passed in
    Returns:
        Prints the output and returns None
    """
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0),
    )
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
    return None


def update_metadata_for_texts(document: Document):
    """
    langchain's load_qa_with_sources_chain method requires metadata to contain a
    "source" key so that it can be returned along with results. This function adds that
    key so that the reference to the arxiv paper can be returned

    Args:
        document:
            The Document object returned by the document_loader
    Returns:
        The document object along with the reworked "source" key
    """
    document_new = document.copy()

    # Format the source url so it plays nicely with arXiv and its a clickable url
    source = "https://www.arxiv.org/abs/" + document_new.metadata["file_path"].split(
        "/"
    )[-1][:-4].replace("_", ".")
    document_new.metadata = {"source": source}
    return document_new


if __name__ == "__main__":
    # Initialize an embeddings model.
    # Currently only ada embeddings are available: https://openai.com/pricing
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    # Initialize a pinecone database
    index_name = "research-assistant"
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find on the console at app.pinecone.io
        environment=PINECONE_API_ENV,
    )

    # The if statement is used to save on costs. We won't recalculate vector embeddings
    # if it is not required.

    # More on Document Loaders:
    # https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
    # https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pymupdf
    if EMBED_PAPERS:
        loader = DirectoryLoader(
            path=f"{DIRNAME}/data/papers/batteries/",
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            silent_errors=True,
        )

        papers = loader.load()

        # The separator here is important, or sometimes latex in papers can get so large
        # without splitting that they can break the API call because the request token
        # length can be ginormous
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separator="\n"
        )
        texts = text_splitter.split_documents(papers)
        texts_with_metadata = [update_metadata_for_texts(text) for text in texts]

        search_index = PineconeCustom.from_texts(
            texts=[t.page_content for t in texts_with_metadata],
            embedding=embeddings,
            index_name=index_name,
            metadatas=[t.metadata for t in texts_with_metadata],
        )

    else:
        search_index = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, text_key="text"
        )

    print_answer(
        search_index=search_index,
        question="How does the inclusion of carbon help make batteries more performant?",
    )

    # Alternate implementation to answer question, but doesn't return sources at the
    # monent

    # retriever = docsearch.as_retriever()
    # qa = RetrievalQA.from_chain_type(
    #     llm=OpenAI(), chain_type="stuff", retriever=retriever
    # )

    # query = "What are some high yield ways to make LFPs"
    # qa.run(query)
