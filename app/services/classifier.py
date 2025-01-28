import joblib
import numpy as np
from keras.models import load_model

class ClassifierService:
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.model = None
        self.load_model()

    def load_model(self):
        if self.model_type == "logistic_regression":
            self.model = joblib.load("pretrained/logistic_regression_peace_classifier.joblib")
        elif self.model_type == "neural_network":
            self.model = load_model("pretrained/neural_network.h5")
        elif self.model_type == "random_forest":
            self.model = joblib.load("pretrained/random_forest_peace_classifier.joblib")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def update_model(self, model_type):
        self.model_type = model_type
        self.load_model()

    def classify(self, embedding):
        embedding = np.array(embedding).reshape(1, -1)
        return self.model.predict(embedding).tolist()