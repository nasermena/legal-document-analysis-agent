from pydantic import BaseModel
from typing import List, Literal

class AskRequest(BaseModel):
    question: str
    doc_id: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

class AgentAnalyzeRequest(BaseModel):
    doc_id: str
    question: str
    task_type: Literal["general", "summary", "risk", "compliance"] = "general"


class AgentAnalyzeResponse(BaseModel):
    answer: str
    tool_calls: List[str] = []