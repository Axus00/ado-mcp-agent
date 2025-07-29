from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)