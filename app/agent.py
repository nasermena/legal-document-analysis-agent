from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.rag import retrieve

load_dotenv()


def _get_llm() -> ChatOllama:
    return ChatOllama(
        model="llama3:latest",
        temperature=0,
    )


_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = _get_llm()
    return _llm


def _build_context(doc_id: str, question: str) -> tuple[str, List[str]]:
    docs = retrieve(doc_id, question)
    if not docs:
        return "No relevant clauses were found for this question.", []

    snippets: List[str] = []
    for d in docs[:4]:
        snippets.append(d.page_content.strip())

    context = "\n\n---\n\n".join(snippets)
    return context, snippets

LEGAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a legal document analysis assistant. "
                "You specialize in contract review, risk identification, and compliance checks.\n\n"
                "Always base your answer ONLY on the provided contract context. "
                "If the answer is not in the context, clearly say that.\n\n"
                "Answer clearly and concisely. You may respond in English."
            ),
        ),
        (
            "human",
            (
                "Task type: {task_type}\n"
                "User question: {question}\n\n"
                "Relevant contract context:\n"
                "-------------------------\n"
                "{context}\n"
                "-------------------------\n\n"
                "Using ONLY the context above, provide a clear legal answer. "
                "Quote key clauses when helpful."
            ),
        ),
    ]
)


def run_legal_agent(
    question: str,
    doc_id: str,
    task_type: str = "general",
) -> Dict[str, Any]:
    context, snippets = _build_context(doc_id, question)
    llm = get_llm()
    chain = LEGAL_PROMPT | llm

    response = chain.invoke(
        {
            "task_type": task_type,
            "question": question,
            "context": context,
        }
    )

    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "tool_calls": snippets,
    }