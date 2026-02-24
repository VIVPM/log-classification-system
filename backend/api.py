from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import pandas as pd
import io
import os
import sys
import threading
import traceback
from datetime import datetime
import importlib

# HF Hub Loading capabilities
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "vivpm99/log-classification-model")

def _hf_enabled() -> bool:
    return HF_AVAILABLE and bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"

def pull_model_from_hf():
    """Download best model from HF Hub if available."""
    if not _hf_enabled():
        print("ℹ️ HF Hub not configured or unavailable — skipping pull.")
        return False
        
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"🔄 Pulling model from HuggingFace Hub ({HF_REPO_ID})...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="log_classifier.joblib",
            token=HF_TOKEN,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print("✅ Successfully pulled model from HuggingFace.")
        return True
    except Exception as e:
        error_msg = str(e)
        if "404 Client Error" in error_msg or "EntryNotFoundError" in error_msg or "RepositoryNotFoundError" in error_msg:
            print(f"ℹ️ Model not found on HF Hub ({HF_REPO_ID}). It means you need to train your model first.")
        else:
            print(f"⚠️ Failed to pull model from HF Hub: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: pull model from Hub if we have a token
    pulled = pull_model_from_hf()
    
    # Reload local predict scripts in case the file just changed under them
    if pulled:
        try:
            if 'processor_bert' in sys.modules:
                importlib.reload(sys.modules['processor_bert'])
            if 'classify' in sys.modules:
                importlib.reload(sys.modules['classify'])
        except Exception as e:
            print(f"⚠️ Could not reload modules after HF pull: {e}")
            
    yield
    print("🛑 Shutting down backend API")

import classify
from train_model import run_training_pipeline
from processor_bert import model_classification

app = FastAPI(title="Log Classification API", lifespan=lifespan)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SingleLogRequest(BaseModel):
    source: str
    log_message: str

class TrainingStatusResponse(BaseModel):
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    accuracy: Optional[float]
    num_trained_logs: Optional[int]
    num_features: Optional[int] = None
    message: str
    error: Optional[str]

# ---------------------------------------------------------------------------
# Training status tracking
# ---------------------------------------------------------------------------
training_status = {
    "status": "idle",           # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "accuracy": None,
    "num_trained_logs": None,
    "num_features": None,
    "message": "No training has been run yet.",
    "error": None,
}
training_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Log Classification Backend API"}

@app.get("/model/info")
def get_model_info():
    """Get information about the current model."""
    try:
        from processor_bert import model_classification
        num_features = getattr(model_classification, "n_features_in_", "Unknown")
    except Exception:
        num_features = "Unknown"
        
    return {
        "model_name": "Logistic Regression",
        "num_features": num_features,
    }

@app.post("/predict")
def predict_single(request: SingleLogRequest):
    try:
        label = classify.classify_log(request.source, request.log_message)
        return {
            "source": request.source,
            "log_message": request.log_message,
            "predicted_label": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV format")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns")

        # Process exactly like original classify() structure logic
        logs = list(zip(df["source"], df["log_message"]))
        labels = classify.classify(logs)
        
        df["predicted_label"] = labels
        
        # Return records to frontend for display
        return {"predictions": df.to_dict(orient="records")}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------
def _run_training_pipeline_background(data_path: str):
    global training_status

    with training_lock:
        training_status["status"] = "running"
        training_status["started_at"] = datetime.now().isoformat()
        training_status["completed_at"] = None
        training_status["error"] = None
        training_status["message"] = "Training started... Generating embeddings & training model."

    try:
        # Expected path where processor_bert expects the model to be
        target_model_path = os.path.join(os.path.dirname(__file__), "models", "log_classifier.joblib")
        
        # Run the training pipeline!
        result = run_training_pipeline(data_path, target_model_path)
        
        # Critical: Force a reload of the joblib model inside the processor
        # so subsequent predictions use the *new* model.
        import processor_bert
        importlib.reload(processor_bert)
        importlib.reload(classify) # reload classify to use the new processor_bert

        with training_lock:
            # Update status on success
            training_status["status"] = "completed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"] = "Training completed! Model reloaded successfully."
            training_status["accuracy"] = result["accuracy"]
            training_status["num_trained_logs"] = result["num_trained_logs"]
            training_status["num_features"] = result.get("num_features", None)

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ Training failed:\n{err_msg}")
        with training_lock:
            training_status["status"] = "failed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"] = f"Training failed: {str(e)}"
            training_status["error"] = err_msg

# ---------------------------------------------------------------------------
# Training Trigger & Status Endpoints
# ---------------------------------------------------------------------------
@app.post("/train")
async def trigger_training(file: UploadFile = File(...)):
    """
    Upload the dataset CSV and trigger model retraining in the background.
    """
    global training_status

    # Reject if already running
    with training_lock:
        if training_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="Training is already in progress. Check GET /train/status.",
            )

    # Validate file type
    if not file.filename.endswith((".csv")):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Save uploaded file
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, "training_dataset.csv")
    
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    print(f"📁 Uploaded dataset saved to: {save_path}")

    # Launch training in background thread
    thread = threading.Thread(
        target=_run_training_pipeline_background,
        args=(save_path,),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Training started in background.",
        "status": "running",
        "poll_url": "/train/status",
    }

@app.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Poll training progress. Status: idle | running | completed | failed."""
    with training_lock:
        return TrainingStatusResponse(**training_status)
