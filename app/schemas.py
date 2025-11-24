from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    doc_id: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]