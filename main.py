from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.agents.orchestrator import get_orchestrator
from app.database import get_db, init_db
from app.ingestion import chunk_text, extract_text
from app.models import Document, QueryLog
from app.schemas import (
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)
from app.storage import get_storage_backend
from app.vectorstore import get_vectorstore

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Multi-Agent Document Intelligence Platform",
    description=(
        "Upload documents and ask questions. Answers are produced by a "
        "router -> retriever -> summarizer -> verifier agent pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    storage = get_storage_backend()
    storage_path = storage.save(file.filename, content)

    text = extract_text(file.filename, content)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    document = Document(
        filename=file.filename, storage_path=storage_path, num_chunks=len(chunks)
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    vectorstore = get_vectorstore()
    vectorstore.add_chunks(document.id, chunks)

    return UploadResponse(
        document=DocumentResponse.model_validate(document),
        message=f"Ingested '{file.filename}' into {len(chunks)} chunks.",
    )


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    orchestrator = get_orchestrator()
    result = orchestrator.answer(request.question)

    log = QueryLog(
        question=result["question"],
        route=result["route"],
        answer=result["answer"],
        verified=result["verification_status"],
        sources=",".join(c["chunk_id"] for c in result["sources"]),
    )
    db.add(log)
    db.commit()

    return QueryResponse(
        question=result["question"],
        route=result["route"],
        answer=result["answer"],
        verification_status=result["verification_status"],
        unsupported_claims=result["unsupported_claims"],
        sources=[SourceChunk(**c) for c in result["sources"]],
    )
