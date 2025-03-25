from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

import time


def key(api, path):
    key_dict = {}
    with open(path) as file:
        for line in file:
            key_dict[line.split("=")[0]] = line.split("=")[1]

    return key_dict[api].strip()




if __name__ == "__main__":

    # api_key = os.environ.get("GROQ_API_KEY")
    key_path = r"C:\Users\SaratKarasala\Documents\Projects\Groq\Keys\key.txt"
    api_key = key("groq", key_path)

    start_time = time.time()


    print("Model Load Time", f"{time.time() - start_time}", "Seconds\n")

    # Reading the doc
    start_time = time.time()
    file_path = r"C:\Users\SaratKarasala\Documents\Projects\LLM\BookGPT\data\harry_potter_pdf\Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    pdf_loader = PyPDFLoader(file_path)
    documents = pdf_loader.load()
    print("Vector Store Indexing Time", f"{time.time() - start_time}", "Seconds\n")

    # Splitting the doc into chunks
    start_time = time.time()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # select which embeddings we want to use
    # embeddings = OpenAIEmbeddings(api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # create the vectorestore to use as the index
    db = FAISS.from_documents(documents, embeddings)
    # db = Chroma.from_documents(documents, OpenAIEmbeddings())

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=ChatGroq(model="llama3-70b-8192", api_key=api_key), chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = "what is the book about?"
    result = qa({"query": query})

    print(result)

    index = VectorstoreIndexCreator(
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
        embedding=embeddings,
        vectorstore_cls=FAISS
    ).from_loaders([pdf_loader])

    query = "what is the book about?"
    answer = index.query(llm=ChatGroq(model="llama3-70b-8192", api_key=api_key), question=query, chain_type="stuff")

    print(answer)

