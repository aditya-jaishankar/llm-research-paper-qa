import os
import argparse

import arxiv
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm

from api_keys import OPENAI_TOKEN
from Paper import Paper

tqdm.pandas()

os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN

def retrieve_papers(query: str, max_results: int = 10):
    """
    TODO: Add doc string
    """
    papers = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    ).results()
    return papers


def download_papers(
    query:str="Lithium Ion Phosphate",
    field:str="Batteries"
):
    """
    Downloads papers from the arxiv
    Args:
        query: The search query
        field: The field of science to check for paper relevance
    returns:
        None. Downloads papers and stores to disk
    """
    results = retrieve_papers(query=query, max_results=500)

    # Prompt Engineering principles can probably improve the performance of this zero-shot classifier.
    template = """
        I want you to act as a helpful, skilled and experienced research scientist in the field of {field}. I will provide with an abstract of a peer reviewed research paper, and I would like you to return "Yes" if the abstract is related to {field} research, and "No" if you think the abstract is not related to {field} research. Only return the words "Yes" or "No", without any further explanation.

        Your abstract is "{abstract}".

        Answer:
    """
    gpt3p5 = ChatOpenAI(model_name="gpt-3.5-turbo", request_timeout=30)

    for result in tqdm(results):
        paper = Paper(result, template, field, language_model=gpt3p5)
        paper.update_relevance()
        paper.download_paper()
