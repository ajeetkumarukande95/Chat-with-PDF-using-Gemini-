import streamlit as st
from pdf_utils import get_pdf_text, get_text_chunks
from vector_store import get_vector_store
from conversational_chain import get_conversational_chain, user_input

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
