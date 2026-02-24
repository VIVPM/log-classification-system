import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load key from .env properly
load_dotenv(dotenv_path=".env")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def classify_with_llm(log_msg):
    """
    Generate a variant of the input sentence. For example,
    If input sentence is "User session timed out unexpectedly, user ID: 9250.",
    variant would be "Session timed out for user 9251"
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
        content = response.text
        
        match = re.search(r'<category>(.*)</category>', content, flags=re.DOTALL)
        category = "Unclassified"
        if match:
            category = match.group(1).strip()
            
        return category
    except Exception as e:
        print(f"Gemini LLM Error: {e}")
        return "Unclassified"


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))