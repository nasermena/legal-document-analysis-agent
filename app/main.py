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
            return AskResponse(answer="No relevant text found.", sources=[])

        context = "\n\n".join(d.page_content for d in docs)
        sources = [d.page_content[:200] for d in docs]
        answer = f"Context found:\n{context[:800]}"

        return AskResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, "Failed to answer question")