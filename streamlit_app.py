import streamlit as st
import requests
import pandas as pd
# ---------------------------------------------------------------------------
# Config — API URL from .streamlit/secrets.toml
# ---------------------------------------------------------------------------
if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"].rstrip("/")
else:
    API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Log Classification System",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Labels styling */
    .lbl-security { background: #dc3545; color: white; padding: 6px 16px; border-radius: 12px; font-weight: bold; }
    .lbl-critical { background: #fd7e14; color: white; padding: 6px 16px; border-radius: 12px; font-weight: bold; }
    .lbl-system { background: #0d6efd; color: white; padding: 6px 16px; border-radius: 12px; font-weight: bold; }
    .lbl-user { background: #20c997; color: white; padding: 6px 16px; border-radius: 12px; font-weight: bold; }
    .lbl-http { background: #6f42c1; color: white; padding: 6px 16px; border-radius: 12px; font-weight: bold; }
    .lbl-unclass { background: #adb5bd; color: white; padding: 6px 16px; border-radius: 12px; font-weight: bold; }

    /* Training status cards */
    .status-running {
        background: #0d6efd;
        border: 2px solid #0a58ca;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
    .status-completed {
        background: #157347;
        border: 2px solid #0f5132;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
    .status-failed {
        background: #b02a37;
        border: 2px solid #842029;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper to map labels to CSS classes
def get_label_html(label):
    lbl_lower = str(label).lower()
    if 'security' in lbl_lower: css = 'lbl-security'
    elif 'critical' in lbl_lower or 'error' in lbl_lower: css = 'lbl-critical'
    elif 'system' in lbl_lower: css = 'lbl-system'
    elif 'user' in lbl_lower: css = 'lbl-user'
    elif 'http' in lbl_lower: css = 'lbl-http'
    else: css = 'lbl-unclass'
    return f'<span class="{css}">{label}</span>'

# ---------------------------------------------------------------------------
# Helper: Check API connection
# ---------------------------------------------------------------------------
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=60)  # 60s to survive Render cold start
        return r.status_code == 200, r.json()
    except (requests.ConnectionError, requests.Timeout):
        return False, {}

def get_model_versions():
    try:
        r = requests.get(f"{API_URL}/model/versions", timeout=10)
        if r.status_code == 200:
            return r.json().get("versions", [])
    except Exception:
        pass
    return []

def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📜 Log Classifier")
    st.markdown("---")

    api_ok, root_data = check_api()

    if api_ok:
        st.success("✅ API Connected")
        
        # Version Selection
        versions = get_model_versions()
        
        if not versions:
            st.warning("⚠️ No model versions found on HF Hub. Train a model first.")
            st.session_state["selected_version"] = "main"
        else:
            selected_version = st.selectbox(
                "📂 Select Model Version",
                options=reversed(versions),
                index=0
            )
            st.session_state["selected_version"] = selected_version

        info = get_model_info()
        if info:
            st.markdown(f"**Model:** `{info.get('model_name', 'Loaded')}`")
            st.markdown(f"**Features:** `{info.get('num_features', 'Unknown')}`")
            if 'best_cv_score' in info and info['best_cv_score']:
                st.markdown(f"**Score:** `{info['best_cv_score']:.4f}`")
    else:
        versions = []
        is_local = "localhost" in API_URL
        if is_local:
            st.error("❌ API Offline")
            st.code("cd backend && uvicorn api:app --reload", language="bash")
        else:
            st.warning("⏳ Backend is waking up (Render cold start ~30s)")
            st.caption("Click **Retry** after a few seconds.")
            if st.button("🔄 Retry", use_container_width=True):
                st.rerun()

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. **Train Model** — upload a new dataset
    2. **Single Prediction** — test an individual log message
    3. **Batch Prediction** — upload `test.csv` in bulk
    """)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📜 AI Log Classification System")
st.caption("Classify system logs automatically using Regex, Google Embeddings, and Gemini LLM")
st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏋️ Train Model",
    "🎯 Single Prediction",
    "📊 Batch Prediction",
])

# ======================== TAB 1: Train Model ================================
with tab1:
    st.subheader("🏋️ Retrain the Log Classification Model")
    st.markdown("""
    Upload the dataset (CSV) to retrain the model from scratch.
    The pipeline runs:
    - **Step 1** — Data Preprocessing (Regex Classification & LegacyCRM filtering)
    - **Step 2** — Feature Engineering (Google Embeddings Generation via `gemini-2.5-flash`)
    - **Step 3** — Model Training (Logistic Regression)
    """)

    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to use training.")
    else:
        st.markdown("#### 📁 Upload Training Data")
        st.info("Upload your training dataset containing `source`, `log_message`, and `target_label`.")

        uploaded_train_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="train_uploader"
        )

        col_btn, col_status = st.columns([1, 2])

        with col_btn:
            train_btn = st.button(
                "🚀 Start Training",
                use_container_width=True,
                type="primary",
                disabled=(uploaded_train_file is None),
            )

        if train_btn and uploaded_train_file is not None:
            with st.spinner("Uploading data and starting training..."):
                resp = requests.post(
                    f"{API_URL}/train",
                    files={"file": (uploaded_train_file.name, uploaded_train_file.getvalue(), "text/csv")},
                    timeout=30,
                )
            if resp.status_code == 200:
                st.success("✅ Training started! Monitor progress below.")
                st.session_state["training_triggered"] = True
            elif resp.status_code == 409:
                st.warning("⚠️ Training already in progress.")
                st.session_state["training_triggered"] = True
            else:
                st.error(f"❌ Failed to start training: {resp.text}")

        # ── Training Status Panel ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Training Status")

        status_placeholder = st.empty()
        steps_placeholder = st.empty()
        metrics_placeholder = st.empty()
        refresh_placeholder = st.empty()

        def render_status():
            # Check if there's an active training session
            training_triggered = st.session_state.get("training_triggered", False)
            
            try:
                r = requests.get(f"{API_URL}/train/status", timeout=10)
                if r.status_code != 200:
                    return None
                s = r.json()
            except Exception:
                return None

            status = s.get("status", "idle")
            message = s.get("message", "")

            # If idle and no training triggered, check if we have a loaded model from HF
            if status == "idle" and not training_triggered:
                selected_version = st.session_state.get("selected_version")
                if selected_version and selected_version != "main":
                    # Show loaded model info
                    info = get_model_info()
                    if info:
                        status_placeholder.markdown(
                            f'<div class="status-completed">✅ <strong>Model Loaded from HF Hub</strong><br>Version: {selected_version}</div>',
                            unsafe_allow_html=True
                        )
                        steps_placeholder.markdown("""
                        | Step | Task | Status |
                        |------|------|--------|
                        | 1 | Regex Preprocessing | ✅ |
                        | 2 | Model Embeddings Generator | ✅ |
                        | 3 | Logistic Regression Training | ✅ |
                        """)
                        m1, m2, m3 = metrics_placeholder.columns(3)
                        model_name = info.get("model_name", "")
                        # Extract only the ML model name (remove BERT/LLM part)
                        if " + " in model_name:
                            model_name = model_name.split(" + ")[0]
                        m1.metric("📊 Model", model_name)
                        acc = info.get("best_cv_score")
                        m2.metric("📈 CV Score", f"{acc:.4f}" if acc else "-")
                        m3.metric("🔢 Features", info.get("num_features", "-"))
                        return "loaded"
                
                # No model loaded, show info message
                status_placeholder.info("📂 Upload your CSV file and click **Start Training** to begin.")
                return "idle"

            elif status == "idle":
                status_placeholder.info("💤 No training has been run yet. Upload data and click **Start Training**.")
                return "idle"

            elif status == "running":
                status_placeholder.markdown(
                    f'<div class="status-running">🔄 <strong>Training in Progress</strong><br>{message}</div>',
                    unsafe_allow_html=True
                )
                
                # Step indicators
                msg_lower = message.lower()
                step1 = "✅" if "generating embeddings" in msg_lower or "complete" in msg_lower else "🔄"
                step2 = "✅" if "training model" in msg_lower or "complete" in msg_lower else (
                    "🔄" if "generating embeddings" in msg_lower else "⏳")
                step3 = "✅" if "complete" in msg_lower else (
                    "🔄" if "training model" in msg_lower else "⏳")

                steps_placeholder.markdown(f"""
                | Step | Task | Status |
                |------|------|--------|
                | 1 | Regex Preprocessing | {step1} |
                | 2 | Model Embeddings Generator | {step2} |
                | 3 | Logistic Regression Training | {step3} |
                """)

            elif status == "completed":
                status_placeholder.markdown(
                    f'<div class="status-completed">✅ <strong>Training Complete!</strong><br>{message}</div>',
                    unsafe_allow_html=True
                )
                steps_placeholder.markdown("""
                | Step | Task | Status |
                |------|------|--------|
                | 1 | Regex Preprocessing | ✅ |
                | 2 | Model Embeddings Generator | ✅ |
                | 3 | Logistic Regression Training | ✅ |
                """)
                m1, m2, m3 = metrics_placeholder.columns(3)
                m1.metric("📊 Logs Trained", s.get("num_trained_logs", "-"))
                acc = s.get("accuracy")
                m2.metric("📈 Training Accuracy", f"{acc:.4f}" if acc else "-")
                m3.metric("🔢 Features Used", s.get("num_features", "-"))

            elif status == "failed":
                status_placeholder.markdown(
                    f'<div class="status-failed">❌ <strong>Training Failed</strong><br>{message}</div>',
                    unsafe_allow_html=True
                )
                if s.get("error"):
                    with st.expander("🔍 Error Details"):
                        st.code(s["error"], language="python")

            return status

        # Manual refresh
        current_status = render_status()

        if current_status == "running":
            refresh_placeholder.markdown("*⏳ Training running in background — click Refresh to check progress.*")
            if st.button("🔄 Refresh Status"):
                st.rerun()
        else:
            refresh_placeholder.empty()
            if current_status in ("completed", "failed"):
                if st.button("🔄 Refresh Status"):
                    st.rerun()

# ======================== TAB 2: Single Prediction ==========================
with tab2:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")
    else:
        st.subheader("Enter Log Details")

        col1, col2 = st.columns([1, 2])
        with col1:
            source = st.selectbox(
                "Log Source",
                ["ModernCRM", "AnalyticsEngine", "ModernHR", "BillingSystem", "ThirdPartyAPI", "LegacyCRM"]
            )
        with col2:
            log_message = st.text_area(
                "Log Message", 
                placeholder="e.g. Email service experiencing issues with sending..."
            )

        st.markdown("")
        predict_btn = st.button("🔮 Classify Log", use_container_width=True, type="primary")

        if predict_btn:
            if not versions:
                st.warning("⚠️ **Training Required:** No models are available on Hugging Face Hub. Please go to the **Train Model** tab to train and register your first model!")
                st.stop()
                
            if not log_message.strip():
                st.error("Please enter a log message.")
            else:
                payload = {
                    "source": source,
                    "log_message": log_message
                }

                with st.spinner("Classifying via backend..."):
                    version = st.session_state.get("selected_version", "main")
                    resp = requests.post(f"{API_URL}/predict?version={version}", json=payload)

                if resp.status_code == 200:
                    result = resp.json()

                    st.markdown("---")
                    st.subheader("Prediction Result")

                    r1, r2 = st.columns([3, 1])

                    with r1:
                        st.markdown(f"**Source:** `{result['source']}`")
                        st.markdown(f"**Message:** *{result['log_message']}*")
                    with r2:
                        label = result["predicted_label"]
                        st.markdown(f"<div style='text-align: right;'>{get_label_html(label)}</div>", unsafe_allow_html=True)
                else:
                    st.error(f"API error: {resp.text}")

# ======================== TAB 3: Batch Prediction ===========================
with tab3:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")
    else:
        st.subheader("Upload Log CSV")
        st.info("CSV must contain specifically named columns: **source**, **log_message**")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None and api_ok:
            df_preview = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df_preview)} logs**")
            st.dataframe(df_preview.head(), use_container_width=True)

            if st.button("🔮 Classify All Logs", use_container_width=True, type="primary", key="batch_predict"):
                if not versions:
                    st.warning("⚠️ **Training Required:** No models are available on Hugging Face Hub. Please go to the **Train Model** tab to train and register your first model!")
                    st.stop()
                    
                uploaded_file.seek(0)

                with st.spinner(f"Classifying {len(df_preview)} logs... This might take a moment."):
                    version = st.session_state.get("selected_version", "main")
                    resp = requests.post(
                        f"{API_URL}/predict/batch?version={version}",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    results_df = pd.DataFrame(data["predictions"])

                    st.markdown("---")
                    st.subheader("Results Summary")

                    # Calculate category distribution
                    distribution = results_df["predicted_label"].value_counts()
                    
                    st.bar_chart(distribution)

                    st.subheader("Detailed Predictions")
                    # Add styled HTML classification column for display only
                    display_df = results_df.copy()
                    
                    st.dataframe(display_df, use_container_width=True)

                    csv_out = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Classified Logs CSV",
                        csv_out,
                        "classified_logs.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error(f"API error: {resp.text}")
