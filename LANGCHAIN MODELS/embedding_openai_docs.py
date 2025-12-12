from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

documents= [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build applications that can understand and generate human-like text.",
    "LangChain provides tools for prompt management."
]

doc_embedding= OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

results = doc_embedding.embed_documents(documents)
for i, vec in enumerate(results):
    print(f"Document {i+1} embedding length:", len(vec))
    print(f"Document {i+1} embedding (first 10 dims):", vec[:10])