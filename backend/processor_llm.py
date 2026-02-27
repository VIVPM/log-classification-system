# LLM-based fallback for LegacyCRM logs.
# LegacyCRM generates unstructured business process errors that regex can't match
# and there's not enough labeled data to train the ML model on them. Gemini
# reads the log and picks from two specific categories.

import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def classify_with_llm(log_msg):
    """
    Ask Gemini to classify a LegacyCRM log into one of two categories.
    Parses the response for <category>...</category> tags — if missing or
    ambiguous, returns "Unclassified".
    """
    prompt = f'''Classify the log message into one of these categories:
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags.
    Log message: {log_msg}'''

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5)
        )
        match = re.search(r'<category>(.*)</category>', response.text, flags=re.DOTALL)
        return match.group(1).strip() if match else "Unclassified"
    except Exception as e:
        print(f"Gemini LLM error: {e}")
        return "Unclassified"


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))