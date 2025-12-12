from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file  

embedding= OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result = embedding.embed_query("What is Ollama?")
print("Embedding length:", len(result))
print("Embedding (first 10 dims):", result[:10])