# ------------- First I need to import all of the libraries. So, I am importing them right now ---------------X

# 1. library to create web apps 
import streamlit as st 

# 2. Library to interact with the operating system for reading writing the files
import os 

# 3. Library to interact with PDF files to enable reading and user communication
from PyPDF2 import PdfReader 

# 4. Library which allows us to work with Microsoft Word (.docx) files in Python.
import docx 

# 5. Library which allows user to process .zip file 
import zipfile

# 6. This is used to allow user to interact with the chat gpt open AI model
from langchain.chat_models import ChatOpenAI 

# 7. It interacts with the OpenAI model to let the user communicate and get answers about the pdf 
from langchain.llms import OpenAI 

# 8. It is commonly used to protect the sensitive data of the user
from dotenv import load_dotenv 

# 9. Embedding means to convert textual data into numerical binary form (easily understandable by the computer). Embedding can be considered as keeping the two or more than two things together (who have the same kind of properties for the purupose of searching in a better way)
from langchain.embeddings import HuggingFaceEmbeddings

# 10. A library through which we can store the vector embeddings into the vector data base 
# from elasticsearch import Elasticsearch # I am commeni=ting it , because I am gonna use FAISS Vector DB again 
from langchain.vectorstores import FAISS

# 11. A library through which you can likely incorporating a component related to conversational retrieval 
from langchain.chains import ConversationalRetrievalChain 

# 12. A library for temporarily storing data, acting as a buffer memory
from langchain.memory import ConversationBufferMemory 

# 13. A library, indicating a usage of the Hugging Face model hub for language-related tasks
from langchain import HuggingFaceHub

# 14. Importing the 'message' module for user-model interaction in a chat application
from streamlit_chat import message 

# 15. Importing the 'get_openai_callback' function from the 'callbacks' module in langchain
from langchain.callbacks import get_openai_callback 

# 16. Importing SentenceTransformer, a library for converting sentences into vector embeddings(different things of similar properties with each other)
from sentence_transformers import SentenceTransformer

# 17. Importing the library of 'text splitter' so that I can pass my data into different chunks , As I know that the model has the ability to read 3000 words at a time 
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 18. A library to add the document loader in web app of streamlit
from time import sleep


# X-------------------------------------All Libraries have been imported--------------------------------------X


# Through this code , I am safely and securely using and accessing my API key of Open AI 
openapi_key = st.secrets["OPENAI_API_KEY"]
# Note: From the (.streamlit folder) I have deleted secrets.toml to keep this API key safe and secure at the time of pushing code on my git hub


# Here, 'main' function will get started. Where all major operations will get perform in streamlit web app. 
def main():

    # A Function to Load environment variables from .env file
    load_dotenv()

    st.set_page_config(page_title="Chatting With Files")
    st.header("Owais Document GPT")

    # 'st.session_state' is used to store and retrieve user input across different parts of the Streamlit app.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf' , "docx" , "doc" , 'zip'],accept_multiple_files=True)
        openai_api_key = openapi_key
        process = st.button("Process PDF File")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        st.write("File is loading... Please wait.")
        progress_bar = st.progress(0)  # Initialize a progress bar

        # Simulate loading with a sleep function. A Progress icon will show , which will work like a loader
        for percent_complete in range(0, 101, 10):
            progress_bar.progress(percent_complete)
            sleep(1)  # Simulating loading time


        files_text = get_files_text(uploaded_files)
        st.success("File loaded successfully!")

        # From here , I will start making chunks of my data . So that model can take it easily
        
        text_chunks = get_text_chunks(files_text, use_first_splitter=True)
        st.write("file chunks created.....")


        # Here , I am creating the vector stores. They will help me to allow our application to efficiently search for and retrieve relevant information from the text chunks based on their semantic content (similar embeded content which will be created by embeded model of hugging face)
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vector Store Created... Now Ask Questions from this file")

         # create conversation chain (One information is directly concerned and connected with the other one)
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) #for openAI

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Questions...")
        if user_question:
            handel_userinput(user_question)

# Function to get the input file and read the text from it.
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".zip":
            text += get_zip_file(uploaded_file)
    
    return text

# There is a function which will deal with all the PDF files having externsion of file.pdf
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# There is a function which will deal with all the Text files having externsion of file.doc
def get_docx_text(file):
    # Create a Document object using python-docx
    doc = docx.Document(file)
    
    # Initialize an empty list to store text from paragraphs
    allText = []
    
    # Iterate through paragraphs in the document
    for docpara in doc.paragraphs:
        # Append the text of each paragraph to the list
        allText.append(docpara.text)
    
    # Join the list of text into a single string using ' '.join()
    text = ' '.join(allText)
    
    # Return the final text
    return text

# Function to handle the outputs of my zip files in this document gpt
def get_zip_file(file):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        # Assuming you want to concatenate the text from all files in the ZIP
        text = ""
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as zip_file:
                # Read and decode the content of each file in the ZIP
                file_content = zip_file.read().decode('utf-8')
                text += file_content
    return text

# ---------------------------------------- Here, I am going to create the chunks of my data in this document GPT--------------------------------------- 

# A function to create the chunks of my data using the first text splitter (split by character)
def get_text_chunks_first_splitter(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# A function to create the chunks of my data using the RecursiveCharacterTextSplitter
def get_text_chunks_second_splitter(text):


    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n",
        chunk_size= 500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_chunks(text, use_first_splitter=True):
    if use_first_splitter:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
    else:

        text_splitter = RecursiveCharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )


    # This split text method will split and break the characters into smaller chunks 
    chunks = text_splitter.split_text(text)
    return chunks

# Creating an Elasticsearch instance. Adjust the parameters accordingly.
# es = Elasticsearch([{'host': 'localhost', 'port': 8501, 'scheme': 'https'}])
# # Set a higher timeout value (e.g., 30 seconds)
# es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200 , 'scheme': 'https'}], timeout=30)
# def get_vectorstore(text_chunks, es_instance=es):
#     # Using the Hugging Face embedding model as per the given instructions.
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
#     # Creating an index in Elasticsearch. Replace 'document_embeddings_index' with your desired index name.
#     index_name = 'document_embeddings_index'
#     es_instance.indices.create(index=index_name, ignore=400)
#     # Indexing each text chunk along with its embedding in Elasticsearch
#     for i, chunk in enumerate(text_chunks):
#         # Get the embedding for the text chunk
#         embedding = embeddings.encode(chunk)
#         # Index the document embedding in Elasticsearch
#         es_instance.index(index=index_name, id=i, body={'embedding': embedding.tolist(), 'text_chunk': chunk})
#     # Returning the Elasticsearch instance and the index name
#     return es_instance, index_name

# ----------------------------------------------------- Sorry Sir, I have tried my level best to use the elastic search vector db in my document gpt . But I deliberately got failed in it & we know that pinecone is paid vector db . Therefore, I am going to use FAISS vector Database . So that I can atleast show my document gpt to you in the running state 


def get_vectorstore(text_chunks):
    # Using the hugging face embedding models
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    # creating the Vectore Store using Facebook AI Semantic search
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base



def get_conversation_chain(vetorestore,openai_api_key):

    # As per the given task instructions here we have to use LLM of the gpt-3.5-turbo-16k 
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo-16k',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))


if __name__ == '__main__':
    main()