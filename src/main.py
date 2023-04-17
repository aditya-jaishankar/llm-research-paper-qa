import argparse
from paper_downloader import download_papers
from qa_over_paper_corpus import qa_over_paper_corpus

parser = argparse.ArgumentParser(
    description="""
        Get the task to be performed - either download papers for a search query and
        field, or query over the corpus of downloaded papers
    """
)
parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="Required. Can either be 'download' or 'qa'. If downloading papers, also pass values for --searchq and --field arguments.",
    required=True,
)
parser.add_argument(
    "-s",
    "--searchq",
    type=str,
    help="Search query for downloading papers from the arxiv",
)
parser.add_argument(
    "-f",
    "--field",
    type=str,
    help="Field of study against which to check for downloaded paper relevance. Also used as the path where papers are written, and used for paper retrieval during qa",
)
parser.add_argument(
    "-q",
    "--question",
    type=str,
    help="The question you want to ask over the downloaded paper corpus",
)
args = parser.parse_args()

if ags.task not in ("qa", "download"):
    print(
        "task argument can only take values download or qa. Use python main.py -h for more details"
    )
    SystemExit(1)

if args.task == "download" and (not args.searchq or not args.field):
    print(
        "searchq and field arguments must be provided for downloading papers. Use python main.py -h for more details"
    )
    SystemExit(1)

if args.task == "qa" and (not args.question or not args.field):
    print(
        "Both a question to ask over the corpus and the field is required. The value of field should correspond to where papers were initially downloaded. Use python main.py -h for more details."
    )
    SystemExit(1)


if __name__ == "__main__":
    if args.task == "download":
        download_papers(query=args.searchq, field=args.field)
    elif args.task == "qa":
        qa_over_paper_corpus(question=args.question, field=args.field)
