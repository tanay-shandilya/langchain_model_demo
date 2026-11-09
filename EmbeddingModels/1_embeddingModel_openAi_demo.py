from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

query = 'What is the capital of India?'
query_vector = embeddings.embed_query(query)

# print(query_vector)

# print(len(query_vector))

# print(query_vector[0:6:])


document = [
    "New Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."]

doc_vectors = embeddings.embed_documents(document)

df = pd.DataFrame(doc_vectors)
print(df.shape)

print(df.head)
