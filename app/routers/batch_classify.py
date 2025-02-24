from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.classifier import ClassifierService
from app.services.embeddings import EmbeddingService
from app.services.transcript import TranscriptService

router = APIRouter()
embedding_service = EmbeddingService()
classifier_service = ClassifierService()


class BatchVideoRequest(BaseModel):
    video_urls: List[str]


@router.post("/videos")
async def classify_videos(request: BatchVideoRequest):
    try:
        transcript_service = TranscriptService()
        results = []

        for url in request.video_urls:
            # Get transcript
            transcript = transcript_service.get_transcript(url)
            if not transcript:
                results.append({
                    "url": url,
                    "status": "error",
                    "error": "Failed to fetch transcript"
                })
                continue

            # Generate embedding
            embedding = embedding_service.generate_embedding(transcript)
            #print(embedding)
            # Classify
            classification = classifier_service.classify(embedding)

            results.append({
                "url": url,
                "status": "success",
                "classification": classification
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
