import logging
import pickle
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the preprocessor from the local directory
from preprocessor import preprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spam-classifier-backend")

app = FastAPI(
    title="Email Spam Classifier API",
    description="API for classifying emails as Spam or Ham using a Naive Bayes model.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    processing_time_ms: float
    clean_text: str

def load_models():
    global model, vectorizer
    try:
        logger.info("Loading models...")
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("nb_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # In a real scenario, you might want to trigger a training job here
        # For now, we'll raise an error if someone tries to predict

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "Email Spam Classifier Backend is running.",
        "status": "ready" if model and vectorizer else "models_missing"
    }

@app.get("/health")
async def health():
    if model and vectorizer:
        return {"status": "healthy"}
    return {"status": "unhealthy", "reason": "models not loaded"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not model or not vectorizer:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Please ensure models exist and restart the server."
        )
    
    start_time = time.time()
    
    try:
        # 1. Preprocess
        clean_text = preprocess(request.text)
        
        if not clean_text:
            return PredictResponse(
                label="ham",  # Default to ham for empty/unintelligible text
                confidence=1.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                clean_text=""
            )
            
        # 2. Vectorize
        X = vectorizer.transform([clean_text])
        
        # 3. Predict
        prediction = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        label = "spam" if prediction == 1 else "ham"
        confidence = float(max(probs))
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictResponse(
            label=label,
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time, 2),
            clean_text=clean_text
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
