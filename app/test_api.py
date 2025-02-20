import requests

BASE_URL = "http://127.0.0.1:8000"

def change_model(model_type):
    """Send a request to change the model."""
    url = f"{BASE_URL}/model/update/?model_type={model_type}"
    response = requests.post(url)
    return response.json()

def classify_article(article_text):
    """Classify an article using the current model."""
    url = f"{BASE_URL}/classify/"
    response = requests.post(url, json={"article": article_text})
    return response.json()

if __name__ == "__main__":
    print("Testing FastAPI Model Classification")

    # Change model to Logistic Regression and test
    print("\nSwitching to Logistic Regression Model...")
    print(change_model("logistic_regression"))

    print("\nClassifying with Logistic Regression:")
    article = "Uganda murderer kills 2000"
    print(classify_article(article))

    # Change model to Neural Network and test
    print("\nSwitching to Neural Network Model...")
    print(change_model("neural_network"))

    print("\nClassifying with Neural Network:")
    print(classify_article(article))

    # Change model to Similarity Search and test
    print("\nSwitching to Similarity Search Model...")
    print(change_model("similarity"))

    print("\nClassifying with Similarity Search:")
    print(classify_article(article))
