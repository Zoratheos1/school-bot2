import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from langdetect import detect

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="School AI Assistant",
    page_icon="🎓",
    layout="centered"
)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- CONFIGURATION ---
PDF_FILE_NAME = "School_constitution.pdf"

# --- THE BRAIN (Cached) ---
@st.cache_resource
def setup_brain():
    try:
        # Get Key from Secrets (Hidden)
        api_key = st.secrets["GROQ_API_KEY"]
        
        # Load PDF
        if not os.path.exists(PDF_FILE_NAME):
            st.error("Error: PDF file not found.")
            return None

        loader = PyPDFLoader(PDF_FILE_NAME)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        retriever = vector_db.as_retriever()
        
        # LLM
        llm = ChatGroq(
            temperature=0.1, 
            model_name="llama-3.3-70b-versatile", 
            api_key=api_key
        )

        template = """
        You are a helpful School Assistant.
        Context: {context}
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return qa_chain
    except Exception as e:
        st.error(f"System Error: {e}")
        return None

# --- MAIN INTERFACE ---
st.title("🎓 School Assistant")
st.write("Ask any question about the school rules. I speak English and Uzbek!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching school rules..."):
            chain = setup_brain()
            
            if chain:
                try:
                    # Language Check
                    is_uzbek = False
                    try:
                        if detect(prompt) in ['uz', 'tr', 'az']:
                            is_uzbek = True
                    except: pass

                    # Translate & Ask
                    query = prompt
                    if is_uzbek:
                        query = GoogleTranslator(source='auto', target='en').translate(prompt)

                    response = chain.invoke(query)

                    # Translate Back
                    if is_uzbek:
                        response = GoogleTranslator(source='en', target='uz').translate(response)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error("I'm having trouble connecting right now. Please try again.")
