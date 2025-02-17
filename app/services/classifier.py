import os
import joblib
import numpy as np
from keras.models import load_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

type_to_model_path = {
    "logistic_regression": "pretrained/logistic_regression_peace_classifier.joblib",
    "neural_network": "pretrained/neural_network.h5"
}


class ClassifierService:
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.model = None
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectordb = Chroma(persist_directory="../../db", embedding_function=self.embedding_function)
        self.load_model()

    def load_model(self):
        if self.model_type in type_to_model_path:
            if self.model_type == "logistic_regression":
                self.model = joblib.load(type_to_model_path[self.model_type])
            elif self.model_type == "neural_network":
                self.model = load_model(type_to_model_path[self.model_type])
        elif self.model_type == "similarity":
            self.model = None  # Similarity search does not require a model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def update_model(self, model_type):
        self.model_type = model_type
        self.load_model()

    def classify(self, embedding):
        if self.model_type == "similarity":
            return self.classify_similarity(embedding)
        #embedding = self.embedding_function.embed_query(article_text)
        embedding = np.array(embedding).reshape(1, -1)
        return self.model.predict(embedding).tolist()

    def classify_similarity(self, embedding):
        similar_docs = self.vectordb.similarity_search_by_vector(embedding)
        if not similar_docs:
            return "Uncertain"

        peaceful_count = sum(1 for doc in similar_docs if doc.metadata.get('peaceful', False))
        total_docs = len(similar_docs)
        return "Peaceful" if peaceful_count / total_docs > 0.5 else "Not Peaceful"
