import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class DatabaseService:
    def __init__(self, persist_directory="../../db", embedding_model="text-embedding-3-small"):
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model=embedding_model)
        )

    def query_similar_articles(self, query_text, top_k=5):
        similar_docs = self.vectordb.similarity_search(query_text, k=top_k)

        if not similar_docs:
            return []

        return similar_docs