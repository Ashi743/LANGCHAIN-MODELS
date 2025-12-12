import os
from pathlib import Path
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
print(f"Looking for .env at: {env_path}")
load_dotenv(env_path)

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {bool(api_key)}")

try:
    # Initialize the OpenAI LLM
    llm = OpenAI(model='gpt-3.5-turbo-instruct')
    print("LLM initialized successfully")
    
    # Invoke and print result
    print("Calling OpenAI API...")
    result = llm.invoke("what is langchain?")
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error: {e}") 