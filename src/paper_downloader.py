# Move these functions to the data downloader to keep everything in one place

import os
from pathlib import Path

import pandas as pd

import arxiv
from arxiv import Result
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from tqdm import tqdm

from api_keys import OPENAI_TOKEN, PINECONE_TOKEN
from utils import call_chatgpt_endpoint

tqdm.pandas()

DIRNAME = Path(__file__).resolve().parents[1]

os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN
DIRNAME = Path(__file__).resolve().parent


class Paper:
    def __init__(
        self,
        result: Result,
        template: str,
        field: str,
        abstract: str,
        language_model: ChatOpenAI,
    ):
        """ """
        self.result = result
        self.template = template
        self.input_variables_dict = {"field": field, "abstract": abstract}
        self.language_model = language_model
        self._relevance = None

    
    def is_paper_relevant(self):
        """
        """
        return call_chatgpt_endpoint(
            model=self.language_model,
            template=self.template,
            input_variables_dict=self.input_variables_dict,
        )


    def to_dict(self):
        """
        
        """
        return {
            'result': paper.result,
            'title': paper.result.title,
            'summary': paper.result.summary,
            'is_relevant': paper.relevance
        }

    
    @property
    def relevance(self):
        """
        
        """
        return self._relevance
    
    
    @relevance.setter
    def relevance(self, value):
        self._relevance = value


    def update_relevance(self):
        self.relevance = self.is_paper_relevant()
        print("wait")


def retrieve_papers(query: str, max_results: int = 10):
    """ """
    papers = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,  # Is this is asc or desc order?
    ).results()
    return papers


def download_papers(papers_df:pd.DataFrame, dirpath:str):
    """
    
    """
    papers_df.result.apply(
        lambda result: result.download_pdf(
            dirpath=dirpath,
            filename=(paper.entry_id.split('/')[-1]).replace('.', '_') + '.pdf'
        )
    )
    return None


if __name__ == "__main__":

    results = retrieve_papers(query="Lithium Ion Phosphate", max_results=50)

    template = """
        I want you to act as a helpful, skilled and experienced research scientist in the field of {field}. I will provide with an abstract of a peer reviewed research paper, and I would like you to return "Yes" if the abstract is related to {field} research, and "No" if you think the abstract is not related to {field} research. Only return the words "Yes" or "No", without any further explanation.

        Your abstract is "{abstract}".

        Answer:
    """
    field = "batteries"
    #TODO: Move this to a config?
    gpt3p5 = ChatOpenAI(model_name="gpt-3.5-turbo")
    
    #TODO: Wrap this in a function called get_papers_df
    papers_list = []
    for result in tqdm(results):
        abstract = result.title + '. ' + result.summary
        paper = Paper(result, template, field, abstract, language_model=gpt3p5)
        paper.update_relevance()
        papers_list.append(paper.to_dict())
    
    papers_df = pd.DataFrame(papers_list)
    relevance_filter = papers_df.is_relevant.apply(
        lambda relevance: 'Yes' in relevance
    )

    filtered_papers_df = papers_df[relevance_filter]