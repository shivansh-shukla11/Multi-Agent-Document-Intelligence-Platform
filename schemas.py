from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    route: str
    answer: str
    verification_status: str
    unsupported_claims: List[str] = []
    sources: List[SourceChunk]


class DocumentResponse(BaseModel):
    id: int
    filename: str
    num_chunks: int

    model_config = ConfigDict(from_attributes=True)


class UploadResponse(BaseModel):
    document: DocumentResponse
    message: str
