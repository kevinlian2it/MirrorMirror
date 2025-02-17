import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class DatabaseService:
    def __init__(self, persist_directory="db", embedding_model="text-embedding-3-small"):
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model=embedding_model)
        )

    def query_similar_articles(self, query_text, file_country_code, top_k=5):
        similar_docs = self.vectordb.similarity_search(query_text, k=top_k)

        # Filter out documents from the same country
        filtered_docs = [
            doc for doc in similar_docs
            if doc.metadata.get("country_code", "Unknown") != file_country_code
        ]

        return filtered_docs if filtered_docs else []

    def process_query_csv(self, file_path, file_country_code):
        df = pd.read_csv(file_path, skiprows=range(1, 701), nrows=200)
        df['combined_text'] = df['article_text_Ngram'].str[:1000]

        results = []
        for _, row in df.iterrows():
            query_text = row['combined_text']
            similar_docs = self.query_similar_articles(query_text, file_country_code)

            if similar_docs:
                most_similar_doc = similar_docs[0]
                is_peaceful = most_similar_doc.metadata.get("peaceful", False)
                results.append(is_peaceful)

        return results
