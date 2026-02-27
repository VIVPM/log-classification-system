# Routing logic — decides which of the three classifiers handles each log.
#
# Decision tree:
#   LegacyCRM source → LLM (complex business logic, not enough labeled data for regex/ML)
#   Everything else → regex first (fast, deterministic)
#                     if regex returns None → embed + logistic regression

from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm


def classify(logs):
    """Classify a list of (source, log_message) tuples. Returns a label per entry."""
    return [classify_log(source, msg) for source, msg in logs]


def classify_log(source, log_msg):
    """Route a single log to the right classifier based on source."""
    if source == "LegacyCRM":
        return classify_with_llm(log_msg)

    label = classify_with_regex(log_msg)
    if not label:
        label = classify_with_bert(log_msg)
    return label


def classify_csv(input_file):
    """Read a CSV, classify each row, write output.csv with target_label column."""
    import pandas as pd
    df = pd.read_csv(input_file)
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
    output_file = "output.csv"
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == '__main__':
    classify_csv("test.csv")