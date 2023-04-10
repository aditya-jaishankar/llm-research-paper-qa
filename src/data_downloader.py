from pathlib import Path

import pandas as pd

import arxiv
from arxiv import Result

DIRNAME = Path(__file__).resolve().parents[1]


class Papers():
    
    
    def __init__(self, list_of_papers:list[Result]):
        """
        
        """
        self.papers = list_of_papers
        self.papers_df = pd.DataFrame(self.papers).rename(columns={0: 'arxiv_result'})



# I can run this as a batch job and then embed the abstracts in pinecone
def write_papers_to_disk(query, max_results):
    """
    
    """
    papers = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance  # Is this is asc or desc order?
    ).results()
    
    Path(f'{DIRNAME}/data/papers').mkdir(parents=True, exist_ok=True)
    for paper in papers:
        paper.download_pdf(
            dirpath=f"{DIRNAME}/data/papers",
            filename=(paper.entry_id.split('/')[-1]).replace('.', '_') + '.pdf'
        )
    return None


    """
    1. Download say 50 papers as pdfs using the steps here
    2. Continue along with this tutorial: https://www.mlq.ai/gpt-3-enabled-research-assistant-langchain-pinecone/
        a. Initialize data loaders
        b. Text split for easier retrieval
        c. Initialize embedding model
        d. Initialize pinecone index
        e. Implement QA chain, but more from the perspective of answering knowledge questions about the corpus
           and not so much about search or retrieval, although that might also be useful
        f. Maybe I can convert all this into a streamlit app and then host?
    """

if __name__ == "__main__":
    DOWNLOAD_PAPERS = True
    if DOWNLOAD_PAPERS:
        # write_papers_to_disk(query="Lithium Ion Phosphate Batteries", max_results=500)
        write_papers_to_disk(query="High Energy Physics", max_results=500)