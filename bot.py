import streamlit as st
import os
import traceback # <--- NEW: Helps us see the full error
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

# --- CONFIGURATION ---
st.set_page_config(page_title="School AI", page_icon="🎓")
PDF_FILE_NAME = "School_constitution.pdf"

# 1. SIDEBAR
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    
    if os.path.exists(PDF_FILE_NAME):
        st.success(f"✅ {PDF_FILE_NAME} loaded!")
    else:
        st.error(f"❌ {PDF_FILE_NAME} not found!")

# --- 2. THE BRAIN ---
@st.cache_resource
def setup_brain(groq_key):
    try:
        # Load PDF
        if not os.path.exists(PDF_FILE_NAME):
            raise FileNotFoundError(f"Could not find {PDF_FILE_NAME}")

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
            api_key=groq_key
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
        # --- PRINT THE REAL ERROR ---
        st.error(f"CRITICAL ERROR: {e}")
        st.code(traceback.format_exc()) # Prints the full computer report
        return None

# --- 3. CHAT ---
st.title("🎓 School AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar!")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = setup_brain(api_key)
            
            if chain:
                try:
                    is_uzbek = False
                    try:
                        if detect(prompt) in ['uz', 'tr', 'az']:
                            is_uzbek = True
                    except: pass

                    query = prompt
                    if is_uzbek:
                        query = GoogleTranslator(source='auto', target='en').translate(prompt)

                    response = chain.invoke(query)

                    if is_uzbek:
                        response = GoogleTranslator(source='en', target='uz').translate(response)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error during generation: {e}")
