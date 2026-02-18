# tests/test_model.py
from fastapi.testclient import TestClient
from app.main import app  # Assuming main.py is inside an 'app' folder, adjust if it's in the root

# Create a test client using your FastAPI app
client = TestClient(app)

def test_health_check():
    """Test if the API wakes up and the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_setosa():
    """Test if the model correctly predicts a Setosa flower."""
    # This is the exact same data you just sent via cURL!
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=payload)
    
    # Check that the request was successful
    assert response.status_code == 200
    
    # Check that the prediction matches what we expect
    data = response.json()
    assert data["class_index"] == 0
    assert data["class_name"] == "Setosa"

def test_predict_invalid_data():
    """Test how the API handles missing data."""
    # Missing petal measurements
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
    }
    
    response = client.post("/predict", json=payload)
    
    # FastAPI should automatically reject this with a 422 Unprocessable Entity error
    assert response.status_code == 422