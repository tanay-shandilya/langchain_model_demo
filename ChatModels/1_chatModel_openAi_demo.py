from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatMod = ChatOpenAI(model='gpt-4o-mini', temperature=2, max_completion_tokens=3)

result = chatMod.invoke("What is the capital of India?")

print(result.content)

