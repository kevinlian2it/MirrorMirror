from langchain_openai import OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from app.services.classifier import ClassifierService

router = APIRouter()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/")
class EmbeddingService:
    def __init__(self, model_name="text-embedding-3-small"):
        self.embedding_model = OpenAIEmbeddings(model=model_name)

    def generate_embedding(self, text):
        return self.embedding_model.embed_query(text)
