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
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "vivpm99/log-classification-model")

def _hf_enabled() -> bool:
    return HF_AVAILABLE and bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"

def _load_model_artifacts(version: str = "main", download: bool = True):
    """Download model artifacts from HuggingFace Hub (strict mode)."""
    if not _hf_enabled():
        return False
        
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    if download:
        try:
            api = HfApi(token=HF_TOKEN)
            api.repo_info(repo_id=HF_REPO_ID)
        except Exception as e:
            if "404" in str(e):
                raise ValueError(f"HuggingFace repo '{HF_REPO_ID}' does not exist yet. Please train a model first.")
            raise
            
        print(f"🔄 Pulling model artifacts from HuggingFace Hub ({HF_REPO_ID} @ {version})...")
        
        files_to_download = ["log_classifier.joblib", "model_comparison.csv"]
        for fname in files_to_download:
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=fname,
                    revision=version,
                    token=HF_TOKEN,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✅ Successfully downloaded {fname} (v: {version}).")
            except Exception as e:
                if "model_comparison.csv" in fname and "404" in str(e):
                    print("ℹ️ No model_comparison.csv found, skipping.")
                    continue
                error_msg = str(e)
                if "404 Client Error" in error_msg or "EntryNotFoundError" in error_msg:
                    raise ValueError(f"Model file {fname} not found on HF Hub for version {version}. Train model first.")
                
                raise ValueError(f"Failed to pull {fname} from HF Hub: {e}")
    else:
        print("ℹ️  Download skipped by request — loading from local files directly.")
        
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: pull model from Hub if we have a token
    pulled = False
    try:
        pulled = _load_model_artifacts(version="main")
    except Exception as e:
        print(f"⚠️ Startup model load failed: {e}")
    
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

@app.get("/model/versions")
def get_model_versions():
    """Get available model versions from Hugging Face Hub."""
    if not _hf_enabled():
        return {"versions": ["local"]}
        
    try:
        api = HfApi(token=HF_TOKEN)
        try:
            api.repo_info(repo_id=HF_REPO_ID)
        except Exception as e:
            if "404" in str(e):
                return {"versions": []}
            raise
            
        tags = api.list_repo_refs(repo_id=HF_REPO_ID).tags
        versions = [t.name for t in tags if t.name.startswith("v")]
        sorted_versions = sorted(versions, key=lambda v: tuple(map(int, v.removeprefix('v').split('.'))))
        if not sorted_versions:
            return {"versions": ["main"]}
        return {"versions": sorted_versions}
    except Exception as e:
        print(f"⚠️ Failed to fetch versions from HF Hub: {e}")
        return {"versions": []}

@app.get("/model/info")
def get_model_info():
    """Get information about the current model."""
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "model_comparison.csv")
    if os.path.exists(metrics_path):
        import pandas as pd
        df = pd.read_csv(metrics_path)
        record = df.iloc[0].to_dict()
        return {
            "model_name": record.get("model_name", "Logistic Regression"),
            "num_features": record.get("num_features", 768),
            "best_cv_score": record.get("best_score", None),
            "training_time": record.get("training_time", None)
        }
        
    # Fallback if no csv exists locally
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
def predict_single(request: SingleLogRequest, version: str = "main"):
    try:
        _load_model_artifacts(version=version)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
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
async def predict_batch(file: UploadFile = File(...), version: str = "main"):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV format")
    
    try:
        _load_model_artifacts(version=version)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
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
        
        # Load local model without downloading
        try:
            _load_model_artifacts(download=False)
        except Exception:
            pass
        
        # Critical: Force a reload of the joblib model inside the processor
        # so subsequent predictions use the *new* model.
        import processor_bert
        importlib.reload(processor_bert)
        importlib.reload(classify) # reload classify to use the new processor_bert

        with training_lock:
            # Update status on success
            training_status["status"] = "completed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"] = "Training completed! Model uploaded to HF Hub."
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
