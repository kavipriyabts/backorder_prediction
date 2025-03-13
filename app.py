from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import logging
import uvicorn

# Import your application constants and pipeline classes
from source.constants.application import APP_HOST, APP_PORT
from source.pipeline.prediction_pipeline import PredictionPipeline
from source.pipeline.training_pipeline import TrainPipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI()

# CORS configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route to redirect to documentation
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# Route to train the model
@app.get("/train")
async def train_route_client():
    try:
        logger.info("Training started.")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logger.info("Training completed successfully.")
        return {"message": "Training successful!"}
    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Define request body format for prediction
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all required features based on your dataset

# Route to make predictions
@app.post("/predict")
async def predict_route_client(request: PredictionRequest):
    try:
        logger.info("Prediction started.")
        prediction_pipeline = PredictionPipeline()

        # Ensure request data is passed correctly if needed
        # Since initiate_prediction() does not accept parameters, we do not pass `data`
        result_df = prediction_pipeline.initiate_prediction()  # Call without arguments

        logger.info("Prediction completed successfully.")
        return {"prediction": result_df.to_dict(orient="records")}  # Convert DataFrame to JSON

    except Exception as e:
        logger.error(f"Error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Entry point to run the application
if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=True)


