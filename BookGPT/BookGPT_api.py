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


def query(engine, query):
    return engine.query(query)

def chat(engine, query):
    return engine.chat(query)

def init_llm(llm_model_name, api_key, embedding_model_name):
    llm = Groq(model=llm_model_name, api_key=api_key)
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name) 
    
    Settings.llm = llm
    Settings.embed_model = embed_model


if __name__ == "__main__":

    llm_model_name = "llama3-70b-8192"
    embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
    # "mixedbread-ai/mxbai-embed-large-v1" - 300 seconds
    # "sentence-transformers/all-MiniLM-L6-v2" - 5 seconds
    # "sentence-transformers/all-mpnet-base-v2" - 80 seconds

    # api_key = os.environ.get("GROQ_API_KEY")
    api_key = "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"

    start_time = time.time()    

    print("Model Load Time", f"{time.time() - start_time}", "Seconds\n")
    start_time = time.time()

    documents = SimpleDirectoryReader(input_files=glob.glob("C:/Users/SaratKarasala/Documents/Projects/LLM/BookGPT/data/harry_potter/Harry Potter 1 - Sorcerer's Stone.txt")).load_data()

    index = VectorStoreIndex.from_documents(documents)
    print("Vector Store Indexing Time", f"{time.time() - start_time}", "Seconds\n")
    start_time = time.time()

    query_engine = index.as_query_engine(similarity_top_k=5)

    memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
    chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(),
    memory=memory,
    llm=llm
    )

    print("Testing the RAG system - Query Engine")
    print("----------------------\n")

    questions = ["Can you give me a brief summary of the book?",
                "Give a brief description of the protagonist and antagonist of the book.",
                "Can you give me a complete summary of the story?"]

    for each_Q in questions:
        print("\n", each_Q, "\nResponse:\n")
        response = query(query_engine, each_Q)
        print(response)


    print("Testing the RAG system - Chat Engine")
    print("----------------------\n")

    for each_Q in questions:
        print("\n", each_Q, "\nResponse:\n")
        response = chat(chat_engine, each_Q)
        print(response)
