import uuid
from typing import Any, Callable, List, Optional

from tqdm import tqdm

from langchain import LLMChain, PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings.base import Embeddings


class PineconeCustom(Pinecone):
    def __init__(
        self,
        index: Any,
        embedding_function: Callable,
        text_key: str,
        namespace: Optional[str] = None,
    ):
        super().__init__(
            index=index,
            embedding_function=embedding_function,
            text_key=text_key,
            namespace=namespace
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        text_key: str = "text",
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> Pinecone:
        """
        We redefine the underlying langchain method so that we can add a progress bar
        to the method. See the original langchain implementation of the Pinecone class
        for more details.
        
        Construct Pinecone wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Pinecone index

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Pinecone
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                pinecone = Pinecone.from_texts(
                    texts,
                    embeddings,
                    index_name="langchain-demo"
                )
        """
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )
        _index_name = index_name or str(uuid.uuid4())
        indexes = pinecone.list_indexes()  # checks if provided index exists
        if _index_name in indexes:
            index = pinecone.Index(_index_name)
        else:
            index = None
        for i in tqdm(range(0, len(texts), batch_size)):
            # set end position of batch
            i_end = min(i + batch_size, len(texts))
            # get batch of texts and ids
            lines_batch = texts[i:i_end]
            # create ids if not provided
            if ids:
                ids_batch = ids[i:i_end]
            else:
                ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
            # create embeddings
            embeds = embedding.embed_documents(lines_batch)
            # prep metadata and upsert batch
            if metadatas:
                metadata = metadatas[i:i_end]
            else:
                metadata = [{} for _ in range(i, i_end)]
            for j, line in enumerate(lines_batch):
                metadata[j][text_key] = line
            to_upsert = zip(ids_batch, embeds, metadata)
            # Create index if it does not exist
            if index is None:
                pinecone.create_index(_index_name, dimension=len(embeds[0]))
                index = pinecone.Index(_index_name)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert), namespace=namespace)
        return cls(index, embedding.embed_query, text_key, namespace)
    

def call_chatgpt_endpoint(model, template: str, input_variables_dict: dict):
    """
    TODO: Add docstring
    input_variables is a dict of the form:
        {"field": "batteries", "abstract": "lorem...ipsum}
    for example.
    """
    prompt = PromptTemplate(
        template=template, input_variables=list(input_variables_dict.keys())
    )

    llm_chain = LLMChain(prompt=prompt, llm=model)

    result = llm_chain.run(input_variables_dict)
    return result

