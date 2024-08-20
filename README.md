# Chat-with-PDF-using-Gemini
This Streamlit application allows users to interact with PDF documents using natural language questions. It leverages Google's Gemini Pro model and FAISS for vector search to provide accurate answers based on the content of uploaded PDF files. The application also includes conversational memory, allowing it to remember the context of previous questions.

## Features

- **PDF Text Extraction**: Upload one or more PDF files and extract their text content.
- **Text Chunking**: The extracted text is split into manageable chunks for efficient processing.
- **Vector Store**: Uses FAISS for storing and searching text chunks based on their embeddings.
- **Conversational AI**: Integrated with Google Gemini Pro to generate detailed responses.
- **Conversational Memory**: Remembers the last few interactions for context-aware responses.
- **Interactive UI**: Streamlit-based UI with an easy-to-use interface.

## Installation

### Prerequisites

- Python 3.7 or higher
- Google API key (for using Google Generative AI)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/chat-with-pdf.git
   cd chat-with-pdf
Install dependencies:
Ensure you have pip installed, then run:

pip install -r requirements.txt
Set up environment variables:
Create a .env file in the project root directory and add your Google API key:

GOOGLE_API_KEY=your_google_api_key
Running the Application
Start the Streamlit app:

streamlit run chat_app.py
Upload PDF files:
Use the sidebar to upload your PDF files. Click on "Submit & Process" to start processing the text.

Ask Questions:
Type your questions in the input box on the main screen. The app will display the response, along with a history of the last three interactions.

View Conversational History:
The sidebar will display a record of previous questions and answers, allowing you to follow the conversation's flow.

Customization
Prompt Template: Modify the prompt in the get_conversational_chain function if you need to customize how the model generates responses.
Chunk Size: Adjust the chunk_size and chunk_overlap in the get_text_chunks function to optimize text processing.
Troubleshooting
Ensure your Google API key is valid and has the necessary permissions.
If the app fails to load or process PDFs, check the console output for error messages.
Make sure all dependencies are installed as specified in requirements.txt.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Streamlit
Langchain
Google Generative AI

### Instructions:
- Replace `https://github.com/yourusername/chat-with-pdf.git` with the actual URL of your repository.
- Replace `your_google_api_key` with instructions or a placeholder for the Google API key.
- Modify any sections based on your specific needs or changes you’ve made to the code. 

This `README.md` file should provide users with clear instructions to set up and run your application, along with details about the app's functionality.







