import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up the conversational chain with memory
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Set up memory
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, memory=memory)

    return chain

# Function to handle user input and update the conversation
def user_input(user_question):
    try:
        # Log the user question
        logger.info(f"Processing user question: {user_question}")

        # Load embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logger.info("Embeddings loaded successfully.")

        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully.")

        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        logger.info(f"Similarity search completed. Found {len(docs)} documents.")

        # Get conversational chain and generate response
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        logger.info(f"Generated response: {response['output_text']}")

        # Update conversation history in session state
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        st.session_state["chat_history"].append((user_question, response["output_text"]))

        # Keep only the last three items in the history
        if len(st.session_state["chat_history"]) > 3:
            st.session_state["chat_history"] = st.session_state["chat_history"][-3:]

    except Exception as e:
        # Log the exception
        logger.error(f"An error occurred: {e}")
        # Optionally, you can display an error message to the user
        st.error("An error occurred while processing your request. Please try again later.")

# Main function to set up the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    if "chat_history" in st.session_state:
        st.write("**Conversation History :**")
        for question, answer in st.session_state["chat_history"]:
            st.write(f"**👩‍💼 :** {question}")
            st.write(f"**🤖 :** {answer}")
            st.write("---")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
