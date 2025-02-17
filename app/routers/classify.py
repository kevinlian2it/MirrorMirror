from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.classifier import ClassifierService
from app.services.embeddings import EmbeddingService

router = APIRouter()

embedding_service = EmbeddingService()
classifier_service = ClassifierService()

# Define a Pydantic model to validate incoming JSON payload
class ClassificationRequest(BaseModel):
    article: str

@router.post("/")
def classify_article(request: ClassificationRequest):
    try:
        embedding = embedding_service.generate_embedding(request.article)
        result = classifier_service.classify(embedding)
        return {"classification": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
