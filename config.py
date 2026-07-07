"""
Central configuration, loaded once from environment variables / .env.
Keeping this in one place means every module imports `settings`
instead of scattering os.getenv() calls everywhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM backend
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "offline")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    # Storage
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    LOCAL_STORAGE_DIR: str = os.getenv("LOCAL_STORAGE_DIR", "./storage_data")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./doc_intelligence.db")

    # Vector store
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))


settings = Settings()
