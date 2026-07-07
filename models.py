"""
SQL models. Even though the heavy lifting (semantic search) happens in
the vector store, we keep relational metadata in SQL — this is a
deliberate design choice worth mentioning in an interview: vector DBs
are bad at exact filtering / auditing / relational joins, SQL is good
at exactly that, so we use each for what it's good at.
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    num_chunks = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    route = Column(String)          # "simple" or "complex", set by RouterAgent
    answer = Column(Text)
    verified = Column(String)       # "verified" / "unverified_claims_found"
    sources = Column(Text)          # comma-joined chunk ids used
    created_at = Column(DateTime, default=datetime.utcnow)
