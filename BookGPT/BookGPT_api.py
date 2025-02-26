from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import os
import glob
import time


def answer(engine, query):
    
    return engine.query(query)

def chat(engine, query):
    
    return engine.chat(query)


start_time = time.time()    

# api_key = os.environ.get("GROQ_API_KEY")
api_key = ""

llm = Groq(model="llama3-70b-8192",api_key=api_key)

# "mixedbread-ai/mxbai-embed-large-v1" - 300 seconds
# "sentence-transformers/all-MiniLM-L6-v2" - 5 seconds
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2") 
Settings.llm = llm
Settings.embed_model = embed_model
print("Groq Model Loaded", f"{time.time() - start_time}", " Seconds\n")
start_time = time.time()


documents = SimpleDirectoryReader(input_files=glob.glob("C:/Users/SaratKarasala/Documents/Projects/LLM/BookGPT/data/harry_potter/Harry Potter 1 - Sorcerer's Stone.txt")).load_data()
print("Reader Initialized", f"{time.time() - start_time}", " Seconds\n")
start_time = time.time()

index = VectorStoreIndex.from_documents(documents)
print("Vector Store Index", f"{time.time() - start_time}", " Seconds\n")
start_time = time.time()

query_engine = index.as_query_engine(similarity_top_k=3)

memory = ChatMemoryBuffer.from_defaults(token_limit=5000)
chat_engine = CondensePlusContextChatEngine.from_defaults(
   index.as_retriever(),
   memory=memory,
   llm=llm
)

print("Testing the RAG system\n")
print("Can you give me a brief summary of the book?\n")
response = answer(query_engine, "Can you give me a brief summary of the book?")
print(response)


print("Chat Engine Answers\n")

print("Can you give me a brief summary of the book?\n")
response = chat(chat_engine, "Can you give me a brief summary of the book?")
print(response)

print("Can you tell me a little about the antagonist of the book?\n")
response = chat(chat_engine, "Can you tell me a little about the antagonist of the book?")
print(str(response))

print("Who is the main protagonist of the story and what does he or she do?\n")
response = chat(chat_engine, "Who is the main protagonist?")
print(str(response))
