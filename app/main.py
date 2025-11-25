from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas import AskRequest, AskResponse
from app.rag import ingest_text, retrieve

app = FastAPI(title="Legal Document Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/ingest")
@limiter.limit("10/minute")
async def ingest(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        if not text.strip():
            raise HTTPException(400, "Empty document")

        doc_id = ingest_text(text)
        return {"doc_id": doc_id}

    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, "Failed to ingest document")

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(request: Request, req: AskRequest):
    try:
        docs = retrieve(req.doc_id, req.question)
        if not docs:
            return AskResponse(
                answer="لم يتم العثور على أي جزء مناسب من الوثيقة لهذا السؤال.",
                sources=[],
            )

        # نجهّز المصادر كما هي (أول 200 حرف من كل جزء)
        sources = [d.page_content[:200] for d in docs]

        # -------- 1) نحاول نصنع جواب مختصر من أول جزء --------
        main_doc = docs[0]
        content = main_doc.page_content.replace("\n", " ")
        sentences = [s.strip() for s in content.split(".") if s.strip()]

        if sentences:
            short_answer = ". ".join(sentences[:2])
            direct_answer = f"Direct answer (from contract): {short_answer}."
        else:
            direct_answer = "Direct answer: (could not build a short answer from the retrieved text)."

        # -------- 2) نعرض البنود المرتبطة بشكل مرتب --------
        lines: list[str] = []
        lines.append(f"Question: {req.question}")
        lines.append("")
        lines.append(direct_answer)
        lines.append("")
        lines.append("Relevant clauses (raw text from the contract):")
        lines.append("")

        for idx, d in enumerate(docs[:3], start=1):
            content = d.page_content.strip()
            if not content:
                continue

            content_lines = [l.strip() for l in content.splitlines() if l.strip()]
            heading = content_lines[0] if content_lines else f"Clause {idx}"
            body = " ".join(content_lines[1:])
            body = body[:400]  # نقصّ النص شوي

            lines.append(f"{idx}. {heading}")
            if body:
                lines.append(f"   {body}")
            lines.append("")

        lines.append(
            "Summary: The answer and clauses above are based only on the retrieved parts of the contract."
        )

        answer_text = "\n".join(lines)

        return AskResponse(answer=answer_text, sources=sources)

    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, "Failed to answer question")