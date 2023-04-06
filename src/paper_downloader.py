import os
from pathlib import Path

import arxiv
from arxiv import Result
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm

from api_keys import OPENAI_TOKEN
from utils import call_chatgpt_endpoint

tqdm.pandas()

os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN
DIRNAME = Path(__file__).resolve().parents[1]


class Paper:
    def __init__(
        self,
        result: Result,
        template: str,
        field: str,
        language_model: ChatOpenAI,
    ):
        """ """
        self.result = result
        self.template = template
        self._entry_id = result.entry_id
        self._field = field
        self._relevance = ''
        self._abstract = (
            self.result.title
            + '. '
            + self.result.summary
        ).replace('\n', ' ')
        self.input_variables_dict = {
            "field": self._field,
            "abstract": self._abstract
        }
        self.language_model = language_model

    
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
            'is_relevant': paper._relevance
        }


    def update_relevance(self):
        self._relevance = self.is_paper_relevant()


    def download_paper(self, dirpath=DIRNAME):
        """
        
        """
        write_folder = f"{dirpath}/data/papers/{(self._field).replace(' ', '_')}/"
        Path(write_folder).mkdir(parents=True, exist_ok=True)
        if 'yes' in (self._relevance).lower():
            self.result.download_pdf(
                dirpath=write_folder,
                filename=(paper._entry_id.split('/')[-1]).replace('.', '_') + '.pdf'
            )
        return None
            

def retrieve_papers(query: str, max_results: int = 10):
    """ """
    papers = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    ).results()
    return papers


if __name__ == "__main__":

    results = retrieve_papers(query="Lithium Ion Phosphate", max_results=500)
    
    # Prompt Engineering principles can probably improve the performance of this zero-shot classifier. 
    template = """
        I want you to act as a helpful, skilled and experienced research scientist in the field of {field}. I will provide with an abstract of a peer reviewed research paper, and I would like you to return "Yes" if the abstract is related to {field} research, and "No" if you think the abstract is not related to {field} research. Only return the words "Yes" or "No", without any further explanation.

        Your abstract is "{abstract}".

        Answer:
    """
    field = "batteries"
    gpt3p5 = ChatOpenAI(model_name="gpt-3.5-turbo")

    for result in tqdm(results):
        paper = Paper(result, template, field, language_model=gpt3p5)
        paper.update_relevance()
        paper.download_paper()
    