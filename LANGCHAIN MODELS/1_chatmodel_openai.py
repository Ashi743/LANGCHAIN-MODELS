from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.5 ,max_completion_tokens=10)

response = chat_model.invoke("Hello, write a short poem about AI.")
print(response.content)