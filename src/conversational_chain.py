import streamlit as st
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
