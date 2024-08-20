from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging
logging.basicConfig(level=logging.INFO)

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error reading PDF files: {e}")
        st.error("An error occurred while reading the PDF files.")
    return text

@st.cache
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
