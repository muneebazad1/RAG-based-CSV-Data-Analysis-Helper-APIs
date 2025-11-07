import os

def setup_environment():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = ""   #User your own keys
    os.environ["GOOGLE_API_KEY"] = ""

    print("✅ Environment variables set.")
