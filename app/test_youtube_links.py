import requests

BASE_URL = "http://127.0.0.1:8000"

def change_model(model_type):
    """Send a request to change the model."""
    url = f"{BASE_URL}/model/update?model_type={model_type}"
    response = requests.post(url)
    return response.json()

def classify_videos(video_urls):
    """Classify a batch of videos using the current model."""
    url = f"{BASE_URL}/batch-classify/videos"
    response = requests.post(url, json={"video_urls": video_urls})
    return response.json()

if __name__ == "__main__":
    print("Testing FastAPI Batch Video Classification")

    # Test videos
    peaceful_videos = [
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",  # Happy - Pharrell Williams
        "https://www.youtube.com/watch?v=3AtDnEC4zak"   # The Piano Guys
    ]

    non_peaceful_videos = [
        "https://www.youtube.com/watch?v=DieI-QqcFoQ&list=RDNSDieI-QqcFoQ&start_radio=1",
        "https://www.youtube.com/watch?v=D5_UUx1gdMQ",
        "https://www.youtube.com/watch?v=eDgA4hszFzY"
    ]

    # Test with invalid URL to check error handling
    invalid_videos = [
        "https://www.youtube.com/watch?v=invalid_id",
        "not_a_valid_url"
    ]

    # Test all video types together
    all_videos = peaceful_videos + non_peaceful_videos + invalid_videos

    # Change model to Logistic Regression and test
    print("\nSwitching to Logistic Regression Model...")
    print(change_model("logistic_regression"))

    print("\nClassifying with Logistic Regression:")
    print("Testing batch of all videos:")
    results = classify_videos(all_videos)
    print(results)

    # Change model to Neural Network and test
    print("\nSwitching to Neural Network Model...")
    print(change_model("neural_network"))

    print("\nClassifying with Neural Network:")
    print("Testing peaceful videos only:")
    results = classify_videos(peaceful_videos)
    print(results)

    # Change model to Similarity Search and test
    print("\nSwitching to Similarity Search Model...")
    print(change_model("similarity"))

    print("\nClassifying with Similarity Search:")
    print("Testing non-peaceful videos only:")
    results = classify_videos(non_peaceful_videos)
    print(results)

    # Test error handling with invalid videos
    print("\nTesting error handling with invalid videos:")
    results = classify_videos(invalid_videos)
    print(results)

    # Test empty list handling
    print("\nTesting empty video list:")
    results = classify_videos([])
    print(results)