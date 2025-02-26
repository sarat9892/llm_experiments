from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import os
import glob

# print("Key Loaded: ", os.environ.get("GROQ_API_KEY"))

llm = Groq(model="llama3-70b-8192",api_key="")
embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")  #"sentence-transformers/all-MiniLM-L6-v2"
Settings.llm = llm
Settings.embed_model = embed_model

print("Model Loaded")


documents = SimpleDirectoryReader(input_files=glob.glob("C:/Users/SaratKarasala/Documents/Projects/LLM/BookGPT/data/harry_potter/Harry Potter 1 - Sorcerer's Stone.txt")).load_data()

print("Reader Initialized")

index = VectorStoreIndex.from_documents(documents)
print("vector store index - done")
query_engine = index.as_query_engine(similarity_top_k=3)
print("query engine - done")

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
chat_engine = CondensePlusContextChatEngine.from_defaults(
   index.as_retriever(),
   memory=memory,
   llm=llm
)


print("chat engine created")


response = query_engine.query("What is the book about?")
print(response)


response = query_engine.query("Can you give me a brief summary of the book?")
print(response)


response = chat_engine.chat(
   "Can you tell me a little about the antagonist of the book?"
)
print(str(response))

response = chat_engine.chat(
    "Who is the main protagonist?"
)
print(str(response))

