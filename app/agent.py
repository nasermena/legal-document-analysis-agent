from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.rag import retrieve

load_dotenv()


def _get_llm() -> ChatOllama:
    """
    Create the local LLM used by the legal agent.

    Requires that Ollama is running locally and the model is pulled.
    Example pulled model (as in your machine): `llama3:latest`
    """
    return ChatOllama(
        model="llama3:latest",  # نفس الاسم الظاهر في `ollama list`
        temperature=0,
    )


_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    """Return a singleton LLM instance."""
    global _llm
    if _llm is None:
        _llm = _get_llm()
    return _llm


def _build_context(doc_id: str, question: str) -> tuple[str, List[str]]:
    """
    Use our RAG retriever to get relevant clauses and build a context string.
    Returns (context_text, snippets_list).
    """
    docs = retrieve(doc_id, question)
    if not docs:
        return "No relevant clauses were found for this question.", []

    snippets: List[str] = []
    for d in docs[:4]:
        snippets.append(d.page_content.strip())

    context = "\n\n---\n\n".join(snippets)
    return context, snippets


# إعداد البرومبت اللي رح نستخدمه مع ChatOllama
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
    """
    High-level helper used by the FastAPI endpoint.
    1) يستدعي RAG عشان يجيب الـ context.
    2) يبني برومبت ويدّيه لـ ChatOllama.
    3) يرجّع الجواب + المقاطع المستخدمة كمصدر.
    """
    # 1) نبني الـ context من الـ RAG
    context, snippets = _build_context(doc_id, question)

    # 2) نجهّز الـ chain (prompt -> LLM) وننفّذه
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
        "tool_calls": snippets,  # بدل tool calls الفعلية، نرجّع المقاطع اللي استخدمناها
    }
