# llm-research-paper-qa
An LLM-based system that answers questions about a corpus of research papers from the arXiv. 

In this project, I build a system based on OpenAI's GPT models to answer questions over a corpus of documents. For illustration I use papers downloaded from the arxiv at `https://arxiv.org` and extract content using `langchain`'s `DocumentLoader`, on Lithium Ion Batteries, but you can easily modify this for any content --- journal entires, research notes, movie scripts, anything really.

### Installation

The easiest way to interact with the codebase is to switch to `/src/` and then build the docker image with `docker build -t qa-research-assistant .`, and finally run the built container with `docker run --rm qa-research-assistant  -t TASK -f FIELD -o ORG_ID`. For a full list of arguments, use `docker run --rm qa-research-assistant -h`.
