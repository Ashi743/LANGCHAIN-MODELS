from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "What is LangChain?" 

#documents = [
#    "LangChain is a framework for developing applications powered by language models.",
#    "It enables developers to build applications that can understand and generate human-like text.",
#    "LangChain provides tools for prompt management."
#]

vector = embedding.embed_query(text)  #embed_documents(documents)

print("Embedding length:", len(vector))     #print embedding length
print("Embedding (first 10 dims):", vector[:10])
