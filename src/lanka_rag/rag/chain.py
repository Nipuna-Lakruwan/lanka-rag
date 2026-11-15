"""Answer questions over an index with a local Ollama chat model (LCEL)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from ..embedders import DEFAULT_OLLAMA_HOST
from ..text import LANGUAGE_NAMES, detect_language

DEFAULT_LLM = "qwen3:4b-instruct-2507-q4_K_M"

SYSTEM = """You answer questions about Sri Lankan documents using ONLY the numbered
context passages.
Rules:
- Reply in {language}, even if the passages are in another language.
- Cite the passages you used like [1] or [2][3].
- If the passages do not contain the answer, say you could not find it in the
  documents. Do not guess.
- Keep names, dates, numbers and legal references exactly as written in the passages.
- Be concise: at most 5 sentences or 5 short bullet points. Do not repeat yourself."""

HUMAN = """Context passages:
{context}

Question: {question}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = Path(str(d.metadata.get("source", "?"))).name
        page = d.metadata.get("page")
        where = f"{src}, p.{page}" if page else src
        parts.append(f"[{i}] ({where})\n{d.page_content}")
    return "\n\n".join(parts)


def answer_language(question: str) -> str:
    lang = detect_language(question)
    return LANGUAGE_NAMES.get(lang, "the same language as the question")


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Document]


def build_chain(retriever: BaseRetriever, llm: BaseChatModel):
    """question (str) -> {"answer": str, "sources": [Document]}"""
    retrieve = RunnableParallel(
        question=RunnablePassthrough(),
        sources=retriever,
        language=RunnableLambda(answer_language),
    )
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["sources"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retrieve | RunnablePassthrough.assign(answer=generate)


def make_llm(
    model: str = DEFAULT_LLM, base_url: str = DEFAULT_OLLAMA_HOST, max_tokens: int = 600
) -> ChatOllama:
    # Sinhala costs ~1.3 tokens per character in Qwen/Llama-style tokenizers (about 5x
    # English), so 4 chunks of 800 characters already make a ~4k token prompt. Without
    # an output cap a small model can loop until it overflows the context window.
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.1,
        num_ctx=8192,
        num_predict=max_tokens,
        repeat_penalty=1.15,
    )


def ask(chain, question: str) -> Answer:
    out = chain.invoke(question)
    return Answer(question=question, answer=out["answer"].strip(), sources=out["sources"])
