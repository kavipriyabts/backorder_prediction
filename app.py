from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
import logging

# Import your application constants and pipeline classes
from source.constants.application import APP_HOST, APP_PORT # type: ignore
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
        return Response("Training successful !!")
    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        raise HTTPException(status_code=500, detail=f"Error Occurred! {str(e)}")

# Route to make predictions
@app.get("/predict")
async def predict_route_client():
    try:
        logger.info("Prediction started.")
        prediction_pipeline = PredictionPipeline()
        prediction_pipeline.initiate_prediction()
        logger.info("Prediction completed successfully.")
        return Response("Prediction successful and predictions are stored in S3 bucket !!")
    except Exception as e:
        logger.error(f"Error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error Occurred! {str(e)}")

# Entry point to run the application
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)

