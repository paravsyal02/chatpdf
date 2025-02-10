import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from io import BytesIO

# Initialize Hugging Face's question-answering pipeline
qa_pipeline = pipeline("question-answering")

def get_pdf_text(pdf_files):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf_file in pdf_files:
        # Check if the file has content before processing
        if pdf_file.size > 0:
            # Rewind the file before reading
            pdf_file.seek(0)  # Move pointer to the start of the file
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))  # Open file from BytesIO
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            st.warning(f"Empty file: {pdf_file.name} was skipped.")
    return text

def get_text_chunks(text):
    """Split the text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_answer_from_chunks(question, text_chunks):
    """Retrieve answers from text chunks."""
    answers = []
    for chunk in text_chunks:
        # Pass each chunk and the question to the QA model
        answer = qa_pipeline(question=question, context=chunk)
        answers.append(answer['answer'])
    return answers

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("Chat with PDF using Hugging Face")
    
    # Ask the user for a question
    user_question = st.text_input("Ask a question from the PDF files")
    
    # If the user asks a question
    if user_question:
        # Get the uploaded PDF files from session state
        pdf_docs = st.session_state.get("pdf_docs", [])
        if pdf_docs:
            # Retrieve answers from the chunks
            raw_text = get_pdf_text(pdf_docs)
            if raw_text:  # Check if there's any text extracted
                text_chunks = get_text_chunks(raw_text)
                answers = get_answer_from_chunks(user_question, text_chunks)
                if answers:
                    st.write(f"Answer(s): {', '.join(answers)}")
                else:
                    st.write("Answer is not available in the context.")
            else:
                st.write("No text was extracted from the uploaded PDFs.")
        else:
            st.write("No PDFs uploaded yet.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if pdf_docs:
            st.session_state["pdf_docs"] = pdf_docs  # Store PDFs in session state
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process and split the PDF text into chunks
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        st.success("Done")
                    else:
                        st.warning("No text found in the uploaded PDFs.")
                else:
                    st.warning("No files uploaded.")

if __name__ == "__main__":
    main()
