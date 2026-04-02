# Routing logic — decides which of the three classifiers handles each log.
#
# Decision tree (confidence-based, no source dependency):
#   1. Regex — fast, deterministic for obvious patterns
#   2. BERT embeddings + Logistic Regression — if regex returns nothing
#   3. Gemini LLM — fallback when ML confidence < 0.5

from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm


def classify(logs):
    """Classify a list of log messages. Returns a label per entry."""
    return [classify_log(msg) for msg in logs]


def classify_log(log_msg):
    """Route a single log through regex → BERT → LLM based on confidence."""
    label = classify_with_regex(log_msg)
    if label:
        return label

    label = classify_with_bert(log_msg)
    if label != "Unclassified":
        return label

    return classify_with_llm(log_msg)


def classify_csv(input_file):
    """Read a CSV, classify each row, write output.csv with target_label column."""
    import pandas as pd
    df = pd.read_csv(input_file)
    df["target_label"] = classify(df["log_message"].tolist())
    output_file = "output.csv"
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == '__main__':
    classify_csv("test.csv")