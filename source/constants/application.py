from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import uvicorn

# Ensure the path to the prediction pipeline is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from source.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

# Define request body format
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all required features based on your dataset

@app.get("/")
def home():
    return {"message": "Backorder Prediction API is running on Render!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        pipeline = PredictionPipeline()
        data = request.dict()  # Convert request to dictionary
        result = pipeline.initiate_prediction(data)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("your_script_name:app", APP_HOST="0.0.0.0", APP_PORT=8000, reload=True)


