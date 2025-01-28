from langchain_openai import OpenAIEmbeddings
import openai
import os

class EmbeddingService:
    def __init__(self, model_name="text-embedding-3-small"):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.embedding_model = OpenAIEmbeddings(model=model_name)

    def generate_embedding(self, text):
        return self.embedding_model.embed_query(text)