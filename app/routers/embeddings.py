from fastapi import APIRouter, HTTPException
from app.services.embeddings import EmbeddingService

router = APIRouter()

embedding_service = EmbeddingService()

@router.post("/generate")
def generate_embedding(text: str):
    try:
        embedding = embedding_service.generate_embedding(text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
