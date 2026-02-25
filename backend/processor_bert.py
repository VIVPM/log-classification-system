import joblib
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Use proper path to .env file to avoid OSError
load_dotenv(dotenv_path=".env")

# HF Hub Loading capabilities
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

model_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "log_classifier.joblib")
model_classification = None

def load_or_pull_model():
    global model_classification
    
    # Try pulling from HF if enabled
    if HF_AVAILABLE and HF_TOKEN and HF_TOKEN != "your_hf_token_here" and HF_REPO_ID:
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="log_classifier.joblib",
                token=HF_TOKEN,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            model_classification = joblib.load(downloaded_path)
            return
        except Exception as e:
            error_msg = str(e)
            if "404 Client Error" in error_msg or "EntryNotFound" in error_msg or "RepositoryNotFound" in error_msg:
                print(f"ℹ️ Model not found on HF Hub. You need to train your model first.")
            else:
                pass # Continue to local fallback
                
    # Fallback to local
    if os.path.exists(model_path):
        model_classification = joblib.load(model_path)
    else:
        model_classification = None

load_or_pull_model()

def get_embedding(log_message):
    """
    Generate Google Embeddings for a given text.
    Shared function used by both training and classification.
    """
    try:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=log_message,
            config=types.EmbedContentConfig(task_type="CLASSIFICATION", output_dimensionality=768)
        )
        return [result.embeddings[0].values]
    except Exception as e:
        print(f"Google Embedding Error: {e}")
        return None

def classify_with_bert(log_message):
    try:
        if model_classification is None:
            return "Unclassified"
            
        # Get embeddings from Google
        embeddings = get_embedding(log_message)
        if embeddings is None:
             return "Unclassified"
        
        probabilities = model_classification.predict_proba(embeddings)[0]
        if max(probabilities) < 0.5:
            return "Unclassified"
        predicted_label = model_classification.predict(embeddings)[0]
        
        return predicted_label
    except Exception as e:
        print(f"Google Embedding Error: {e}")
        return "Unclassified"


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)