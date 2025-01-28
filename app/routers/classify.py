from fastapi import APIRouter, HTTPException
from app.services.classifier import ClassifierService
from app.services.embeddings import EmbeddingService

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
classifier_service = ClassifierService()

@router.post("/")
def classify_article(article: str):
    try:
        embedding = embedding_service.generate_embedding(article)
        classification = classifier_service.classify(embedding)
        return {"classification": classification}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
