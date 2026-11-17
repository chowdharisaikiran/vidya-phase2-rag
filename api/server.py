
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag import ask_vidya, create_vector_store

app = FastAPI(
    title="Vidya RAG API",
    description="AI-powered personalized learning assistant",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Vidya RAG API is running!",
        "status": "success"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest_documents():
    try:
        chunks = create_vector_store()

        return {
            "message": "Documents indexed successfully",
            "chunks_indexed": chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        answer = ask_vidya(question)

        source_marker = "**Knowledge Base Sources:**"

        if source_marker in answer:
            answer_text, source_text = answer.split(
                source_marker, 1
            )
            sources = [
                item.strip()
                for item in source_text.split(",")
                if item.strip()
            ]
        else:
            answer_text = answer
            sources = []

        return {
            "question": question,
            "answer": answer_text.strip(),
            "sources": sources,
            "source_type": (
                "knowledge_base" if sources else "general_ai"
            ),
            "status": "success"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
