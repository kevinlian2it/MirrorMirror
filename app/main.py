from fastapi import FastAPI
from app.routers import classify, embeddings, model_config, batch_classify
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Include routers
app.include_router(classify.router, prefix="/classify", tags=["Classification"])
app.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
app.include_router(model_config.router, prefix="/model", tags=["Model Configuration"])
app.include_router(batch_classify.router, prefix="/batch-classify", tags=["Batch Classification"])

@app.get("/")
def root():
    return {"message": "Welcome to the Article Classification API"}
