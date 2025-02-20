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
    "neural_network": "pretrained/ff_model.keras"
}

class ClassifierService:
    _instance = None  # Singleton instance

    def __new__(cls, model_type="logistic_regression"):
        if cls._instance is None:
            cls._instance = super(ClassifierService, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, model_type="logistic_regression"):
        if self.__initialized:
            return
        self.__initialized = True

        self.model_type = model_type
        self.model = None
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectordb = Chroma(persist_directory="db", embedding_function=self.embedding_function)
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
        if model_type != self.model_type:
            self.model_type = model_type
            self.load_model()

    def classify(self, embedding):
        embedding = np.array(embedding).reshape(1, -1)

        if self.model_type == "similarity":
            return self.classify_similarity(embedding)
        elif self.model_type == "logistic_regression":
            prob = self.model.predict_proba(embedding)[0][1]  # Get probability of peaceful class
        elif self.model_type == "neural_network":
            prob = float(self.model.predict(embedding)[0][0])  # Neural network output
        else:
            raise ValueError("Unsupported classification method")

        return {
            "classification": "Peaceful" if prob > 0.5 else "Not Peaceful",
            "probability": prob,
            "model": self.model_type,
        }

    def classify_similarity(self, embedding):
        similar_docs = self.vectordb.similarity_search_by_vector(embedding)
        #print(self.vectordb.get())
        if not similar_docs:
            return {"classification": "Uncertain", "ratio": 0.0}

        peaceful_count = sum(1 for doc in similar_docs if doc.metadata.get('peaceful', False))
        total_docs = len(similar_docs)
        peace_ratio = peaceful_count / total_docs
        return {"classification": "Peaceful" if peace_ratio > 0.5 else "Not Peaceful",
                "probability": peace_ratio,
                "model": self.model_type}

# Create a singleton instance
classifier_service = ClassifierService()
