from fastapi import APIRouter, HTTPException
from app.services.classifier import ClassifierService

router = APIRouter()

classifier_service = ClassifierService()

@router.post("/update")
def update_model(model_type: str):
    try:
        classifier_service.update_model(model_type)
        return {"message": f"Model updated to {model_type}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
