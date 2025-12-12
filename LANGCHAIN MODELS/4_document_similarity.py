from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity
import numpy as np

load_dotenv()  # Load environment variables from .env file

documents= [
    "LangChain is a framework for developing applications powered by language models.",
    "language models are created by OpenAI.",
    "It enables developers to build applications that can understand and generate human-like text.",
    "LangChain provides tools for prompt management."
]

doc_embedding= OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

query= "What is LangChain made by?"

doc_vectors = doc_embedding.embed_documents(documents)
query_vector = doc_embedding.embed_query(query)


# Compute cosine similarities between the query vector and document vectors
cosine_similarities = cosine_similarity([query_vector], doc_vectors)[0] #query vector as 2D array as doc already in 2D
# Get the index of the most similar document
ranked = sorted(list(enumerate(cosine_similarities)), key=lambda x: x[1], reverse=True)

print("Cosine Similarities:", cosine_similarities)
print("Ranked:", ranked)

best_index, best_score = ranked[0]

print("Query:", query)
print("Most similar doc:", documents[best_index])
print("Score:", best_score)