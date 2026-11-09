from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"do_sample": True, "temperature": 0.5, "max_new_tokens": 100},
)
chatMod = ChatHuggingFace(llm=llm)

result = chatMod.invoke("What is the capital of India?")

print(result.content)

