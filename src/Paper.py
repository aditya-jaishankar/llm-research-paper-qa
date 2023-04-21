from pathlib import Path

from arxiv import Result
from langchain.chat_models import ChatOpenAI

from utils import call_chatgpt_endpoint

DIRNAME = Path(__file__).resolve().parents[1]


class Paper:
    """
    Defined a Paper structure with some methods on top of it to download, prioritize, and transform papers downloaded from the arxiv
    """

    def __init__(
        self,
        result: Result,
        template: str,
        field: str,
        language_model: ChatOpenAI,
    ):
        """
        Args:
            result:
                The Result object returned by the arxiv call
            template:
                The engineering Prompt that will be used to determine if the downloaded
                paper is related to the field argument (defined below). To do this, the
                abstract of the paper is passed into a ChatGPT call and a boolean
                question about relatedness to field is passed.
            field:
                The name of the field against which to check if the abstract is related.
                For example, we could check if the downloaded abstract is related to
                field="batteries"
            language_model:
                Typically an openAI Chat language model, such as gpt-3.5-turbo so that
                a question about relevance can be asked, and a boolean yes/no answer is
                returned.
        """
        self.result = result
        self.template = template
        self._entry_id = result.entry_id
        self._field = field
        self._relevance = ""
        self._abstract = (self.result.title + ". " + self.result.summary).replace(
            "\n", " "
        )
        self.input_variables_dict = {"field": self._field, "abstract": self._abstract}
        self.language_model = language_model
        self._is_high_impact = None

    def is_paper_relevant(self):
        """
        Check if the paper abstract is relevant to `field`.
        """
        return call_chatgpt_endpoint(
            model=self.language_model,
            template=self.template,
            input_variables_dict=self.input_variables_dict,
        )

    def to_dict(self):
        """
        Convert the result object into a dictionary for downstream processing
        """
        return {
            "result": self.result,
            "title": self.result.title,
            "summary": self.result.summary,
            "is_relevant": self._relevance,
        }

    def update_relevance(self):
        """
        Update the `_relevance` property by receiving boolean response from chatgpt
        endpoint
        """
        self._relevance = self.is_paper_relevant()

    def download_paper(self, dirpath=DIRNAME):
        """
        Download a relevant papers from the arxiv
        """
        write_folder = f"{dirpath}/data/papers/{(self._field).replace(' ', '_')}/"
        Path(write_folder).mkdir(parents=True, exist_ok=True)
        if "yes" in (self._relevance).lower():
            self.result.download_pdf(
                dirpath=write_folder,
                filename=(self._entry_id.split("/")[-1]).replace(".", "_") + ".pdf",
            )
        return None

    def is_paper_high_impact(self):
        """
        Experimental function that tries to ask if a a given paper is "high impact"
        i.e if the paper is worth reading. This was a stretch to begin with without
        additional work since the gpt model does not have nuanced or expert info about
        scientific importance, but it was worth a shot.

        Without this additional work (i.e. fine-tuning perhaps?), it is best to treat the gpt models as a skilled retrieval librarian rather than a scientific
        researched trained in identifying nuance
        """
        template = """
            I have a paper and I want to know if it is novel and high impact. A high impact paper is defined as one that contains novel work, and is worth my time to read, given competing priorities on my time. I am a skilled scientific researcher in the field of batteries, and as such understand the technical and scientific details of the field. I will provide you the abstract of the paper, and I'd like you to let me know if you think this paper is worth reading. 
            
            In your response, if you think the paper is worth reading, only return the word 'Yes'. Do not provide any explanation. 
            
            If you think the paper is not worth reading, return the word "No" along with an explanation of why you think so. 

            The abstract is {abstract}.
        """
        input_variables_dict = {"abstract": self._abstract}
        return call_chatgpt_endpoint(
            model=self.language_model,
            template=template,
            input_variables_dict=input_variables_dict,
        )

    def update_impact(self):
        """
        Updates the _is_high_impact property. This is experimental --- see the notes for the `is_paper_high_impact()` module for caveats and further thoughts.
        """
        self._is_high_impact = self.is_paper_high_impact()
        return None
