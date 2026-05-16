"""RAG pipeline: retrieve context and generate answers using Hugging Face."""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
from embedding_store import get_vector_store


RAG_SYSTEM_TEMPLATE = """Use the following context to answer the user's question.
If you don't know the answer based on the context, say "I don't have enough information to answer that."

Context:
{context}"""


def get_llm() -> ChatHuggingFace:
    """Initialize the Hugging Face chat model."""
    if not config.HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not set. Add it to your .env file. "
            "See SETUP.md for instructions."
        )

    endpoint = HuggingFaceEndpoint(
        model=config.HF_LLM_MODEL,
        huggingfacehub_api_token=config.HF_TOKEN,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        max_new_tokens=1024,
        task="conversational",
    )

    return ChatHuggingFace(llm=endpoint)


def format_docs(docs):
    """Join retrieved document contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(llm=None, vector_store=None):
    """Build the RAG chain using LCEL."""
    if llm is None:
        llm = get_llm()
    if vector_store is None:
        vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": config.TOP_K_RESULTS},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(question: str, chain_and_retriever) -> dict:
    """Ask a question and get an answer with sources."""
    chain, retriever = chain_and_retriever
    source_docs = retriever.invoke(question)

    answer = chain.invoke(question)

    return {
        "answer": answer,
        "source_documents": source_docs,
    }
