import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import login
from langchain.llms import CTransformers

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=huggingface_token)

# Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

pdf_texts = load_pdf("D:/my_learnings/physics_pdf_demo.pdf")

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
documents = text_splitter.split_documents(pdf_texts)

# Create embeddings
embeding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = FAISS.from_documents(documents, embeding_model)
vector_db.save_local('faiss_index')
vector_db = FAISS.load_local('faiss_index', embeding_model, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever()
llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define function for answering questions
def ask_question(query):
    response = qa_chain.run(query)
    return response
