from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import numpy as np
import os

app = FastAPI(title="Iris Classification API", version="1.0")

# 1. Updated Pydantic V2 syntax (Fixes the warnings!)
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    )

# 2. Fix the path issue! Make it absolute relative to this file's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: model.pkl not found at {MODEL_PATH}. Run train.py first.")

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

@app.post("/predict")
def predict_iris(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    prediction = model.predict(input_data)
    class_index = int(prediction[0])
    
    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}