import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingest import ingest_pdf
from app.rag import ask_question

app = FastAPI(title="AI Document Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Document Assistant is running"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    count = ingest_pdf(file_path)
    return {"message": f"Successfully ingested '{file.filename}'", "chunks": count}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = ask_question(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])
