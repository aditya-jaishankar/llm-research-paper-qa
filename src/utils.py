from langchain import LLMChain, PromptTemplate


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