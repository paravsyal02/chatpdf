import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO  # Ensure to handle BytesIO
import pickle

# Load the environment variables from the .env file
load_dotenv()

# Get the API key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optionally, set it in the environment
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_vector_store(embeddings):
    try:
        # Load the FAISS index from local storage with the correct embeddings
        vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        # If no existing index, initialize a new FAISS object with the embeddings function
        vector_store = FAISS(embedding_function=embeddings.embed_documents, index=None, docstore=None, index_to_docstore_id=None)
    return vector_store

# Helper function to save vector store
def save_vector_store(vector_store):
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vector_store, f)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))  # Correct handling of BytesIO
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def update_vector_store(text_chunks):
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    # Load or initialize the vector store with embeddings
    vector_store = load_vector_store(embeddings)
    # Add the new chunks to the vector store
    vector_store.add_texts(text_chunks)
    # Save updated vector store locally
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make
    sure to provide all the details, if the answer is not in the context provided
    just say "answer is not available in the context" don't give the wrong answer.\n\n 
    Context: \n {context} \n
    Question: \n {question} \n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GEMINI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, selected_pdf):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    
    # Loading the vector store with the embeddings provided
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with PDF using Gemini")

    # Load existing PDF files from storage
    pdf_dir = 'pdf_storage'
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    pdf_options = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if pdf_options:
        selected_pdf = st.selectbox("Select a PDF to ask questions from:", pdf_options)
        user_question = st.text_input("Ask a question from the selected PDF")
        
        if user_question and selected_pdf:
            with open(os.path.join(pdf_dir, selected_pdf), 'rb') as f:
                raw_text = get_pdf_text([f])
            text_chunks = get_text_chunks(raw_text)
            update_vector_store(text_chunks)
            user_input(user_question, selected_pdf)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on the Submit button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    for pdf in pdf_docs:
                        pdf_name = pdf.name
                        with open(os.path.join(pdf_dir, pdf_name), 'wb') as f:
                            f.write(pdf.getbuffer())
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    update_vector_store(text_chunks)
                    st.success("PDFs have been added and processed!")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()