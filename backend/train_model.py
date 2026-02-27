# Training pipeline for the ML leg of the hybrid classifier.
#
# Steps:
#   1. Load the labeled dataset
#   2. Strip logs that regex already handles (they don't need a model)
#   3. Strip LegacyCRM logs (those go to the LLM at inference)
#   4. Get Gemini embeddings for what's left
#   5. Train logistic regression on those embeddings
#   6. Save model + metrics, upload to HF Hub

import pandas as pd
import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from google import genai
from google.genai import types
from processor_regex import classify_with_regex
from processor_bert import get_embedding

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")


def _hf_enabled() -> bool:
    return HF_AVAILABLE and bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"


def upload_to_hf(model_dir: str, accuracy: float):
    """Upload trained artifacts to HF Hub and tag the release. Skips silently if not configured."""
    if not _hf_enabled():
        print("⚠️ HF Hub not configured — skipping upload.")
        return False
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)

        # Figure out the next version tag
        tags     = api.list_repo_refs(repo_id=HF_REPO_ID).tags
        versions = [t.name for t in tags if t.name.startswith("v")]
        if versions:
            latest = sorted(versions)[-1]
            try:
                major, minor = latest.removeprefix("v").split(".")
                new_version  = f"v{major}.{int(minor) + 1}"
            except ValueError:
                new_version = f"v1.{len(versions)}"
        else:
            new_version = "v1.0"

        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=HF_REPO_ID,
            commit_message=f"Training run — {new_version} (acc={accuracy:.4f})",
            token=HF_TOKEN,
            allow_patterns=["*.joblib", "*.json", "*.csv", "*.txt"]
        )
        print(f"☁️ Uploaded artifacts → {HF_REPO_ID}")

        api.create_tag(
            repo_id=HF_REPO_ID,
            tag=new_version,
            tag_message=f"Accuracy: {accuracy:.4f}",
            token=HF_TOKEN
        )
        print(f"🏷️ Tagged as {new_version}")
        return True
    except Exception as e:
        print(f"❌ HF upload failed: {e}")
        return False


def run_training_pipeline(csv_path: str, model_save_path: str):
    """
    Full training pipeline. Loads the CSV, filters down to only the logs
    that the ML model needs to handle, embeds them with Gemini, trains LR.

    Returns a dict with accuracy and dataset stats.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    required = {'source', 'log_message', 'target_label'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    print(f"Total logs loaded: {len(df)}")

    # Strip anything regex can already handle — no point training on those
    print("Applying regex filter...")
    df['regex_label'] = df['log_message'].apply(classify_with_regex)
    df_non_regex      = df[df['regex_label'].isnull()].copy()
    print(f"After regex filter: {len(df_non_regex)} logs remain")

    # Strip LegacyCRM — those go to the LLM at inference time
    df_train = df_non_regex[df_non_regex['source'] != 'LegacyCRM'].copy()
    print(f"After LegacyCRM filter: {len(df_train)} logs for embedding")

    if len(df_train) == 0:
        raise ValueError("No logs left to train on after filtering. Check your dataset.")

    # Embed with Gemini — do one at a time to stay under rate limits
    print("Generating embeddings (gemini-embedding-001)...")
    X_texts  = df_train['log_message'].tolist()
    y_labels = df_train['target_label'].values
    embeddings = []

    for i, text in enumerate(X_texts):
        if i % 100 == 0:
            print(f"  {i}/{len(X_texts)} embeddings done...")
        emb = get_embedding(text)
        if emb:
            embeddings.append(emb[0])
        else:
            # One retry after a longer wait — if it still fails, crash loudly
            time.sleep(15)
            emb = get_embedding(text)
            if emb:
                embeddings.append(emb[0])
            else:
                raise ValueError(f"Embedding failed for: {text[:50]}...")

    print("Embeddings done.")

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, y_labels)

    accuracy = clf.score(embeddings, y_labels)
    print(f"Training accuracy: {accuracy:.4f}")

    print(f"Saving model to {model_save_path}...")
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, model_save_path)

    # Save metrics alongside the model for the HF Hub upload
    features   = clf.n_features_in_ if hasattr(clf, 'n_features_in_') else 768
    metrics_df = pd.DataFrame([{
        "model_name":    "Logistic Regression",
        "best_score":    accuracy,
        "num_features":  features,
        "training_time": pd.Timestamp.now().isoformat()
    }])
    metrics_path = os.path.join(model_dir, "model_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    print("Uploading to HF Hub...")
    upload_to_hf(model_dir=model_dir, accuracy=accuracy)

    return {
        "status":          "success",
        "accuracy":        accuracy,
        "num_trained_logs": len(df_train),
        "num_features":    features
    }
