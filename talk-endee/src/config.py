import os
from dotenv import load_dotenv

load_dotenv()

class Config:
   
    ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080")
    ENDEE_API_BASE = os.getenv("ENDEE_API_BASE", "/api/v1")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    TOP_K = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER != "groq":
            raise ValueError("Only 'groq' provider is supported. Set LLM_PROVIDER=groq or leave unset.")
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env file")
        if not cls.ENDEE_HOST:
            raise ValueError("ENDEE_HOST not set in .env file")
