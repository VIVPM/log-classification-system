import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from google import genai
from google.genai import types
from processor_regex import classify_with_regex

# HF Hub Loading capabilities
try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

def _hf_enabled() -> bool:
    return HF_AVAILABLE and bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"

def upload_to_hf(model_path: str):
    """Upload model artifacts to HuggingFace Hub."""
    if not _hf_enabled():
        print("⚠️ HF Hub not configured — skipping upload.")
        return False
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)
        
        filename = os.path.basename(model_path)
        if os.path.exists(model_path):
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
                token=HF_TOKEN,
            )
            print(f"☁️ Uploaded {filename} → {HF_REPO_ID}")
            return True
    except Exception as e:
        print(f"❌ HF upload failed: {e}")
        return False

def run_training_pipeline(csv_path: str, model_save_path: str):
    """
    Runs the full log classification training pipeline.
    1. Loads dataset
    2. Filters out logs that are classified by regex or LLM (LegacyCRM)
    3. Generates Google Embeddings for remaining logs
    4. Trains Logistic Regression model
    5. Saves the trained model to disk
    """
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'source' not in df.columns or 'log_message' not in df.columns or 'target_label' not in df.columns:
        raise ValueError("CSV must contain 'source', 'log_message', and 'target_label' columns.")

    print(f"Total logs loaded: {len(df)}")

    # 1. Apply regex classification to identify which logs we actually need to train on
    print("Applying regex classification...")
    df['regex_label'] = df['log_message'].apply(classify_with_regex)
    
    # Filter to logs that regex COULD NOT classify
    df_non_regex = df[df['regex_label'].isnull()].copy()
    print(f"Logs remaining after Regex filtering: {len(df_non_regex)}")

    # 2. Filter out LegacyCRM logs (they are handled entirely by the LLM prompt)
    df_non_legacy = df_non_regex[df_non_regex['source'] != 'LegacyCRM'].copy()
    print(f"Logs remaining after LegacyCRM filtering (Target for Embeddings): {len(df_non_legacy)}")

    if len(df_non_legacy) == 0:
        raise ValueError("No logs remaining to train on after regex and LLM filtering. Training aborted.")

    # 3. Generate Google Embeddings
    print("Generating Google Embeddings (models/gemini-embedding-001)...")
    import time
    
    X_texts = df_non_legacy['log_message'].tolist()
    y_labels = df_non_legacy['target_label'].values
    
    embeddings = []
    
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # We batch requests or do them one-by-one to avoid payload limits
    for i, text in enumerate(X_texts):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(X_texts)} embeddings...")
            
        try:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="CLASSIFICATION", output_dimensionality=768)
            )
            embeddings.append(result.embeddings[0].values)
        except Exception as e:
            # Simple retry logic for rate limits
            if '429' in str(e):
                print("  Rate limit hit. Sleeping for 15 seconds...")
                time.sleep(15)
                # Retry once
                result = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(task_type="CLASSIFICATION", output_dimensionality=768)
                )
                embeddings.append(result.embeddings[0].values)
            else:
                raise e

    print("Embeddings generated successfully.")

    # 4. Train Logistic Regression Model
    print("Training LogisticRegression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, y_labels)
    
    accuracy = clf.score(embeddings, y_labels)
    print(f"Training completed. Model accuracy on training set: {accuracy:.4f}")

    # 5. Save the trained model
    print(f"Saving model to {model_save_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    
    # 6. Upload to HF Hub
    print("Uploading model to HuggingFace Hub...")
    upload_to_hf(model_path=model_save_path)
    
    return {
        "status": "success",
        "accuracy": accuracy,
        "num_trained_logs": len(df_non_legacy),
        "num_features": clf.n_features_in_ if hasattr(clf, 'n_features_in_') else 768
    }
