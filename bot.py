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

# --- CONFIGURATION ---
st.set_page_config(page_title="School AI", page_icon="🎓")
PDF_FILE_NAME = "School_constitution.pdf"

# 1. SIDEBAR (API Key & Settings)
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("This AI answers questions based on the School Constitution.")
    
    # Secret Key Input (Safer for websites)
    api_key = st.text_input("Enter Groq API Key:", type="password")
    
    # Check if PDF exists
    if os.path.exists(PDF_FILE_NAME):
        st.success(f"✅ {PDF_FILE_NAME} loaded!")
    else:
        st.error(f"❌ {PDF_FILE_NAME} not found. Please upload it to your folder.")

# --- 2. THE BRAIN (Cached for Speed) ---
@st.cache_resource
def setup_brain(groq_key):
    try:
        # Load PDF
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

        # Chain
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
        return None

# --- 3. THE CHAT INTERFACE ---
st.title("🎓 School AI Assistant")
st.caption("Ask me about the rules! (English & Uzbek supported)")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle New User Input
if prompt := st.chat_input("Ask a question..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Check for API Key
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar first!")
        st.stop()

    # 3. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Load Brain
                chain = setup_brain(api_key)
                if not chain:
                    st.error("Error loading brain.")
                    st.stop()

                # Language Detection
                is_uzbek = False
                try:
                    if detect(prompt) in ['uz', 'tr', 'az']:
                        is_uzbek = True
                except: pass

                # Translate Query
                query = prompt
                if is_uzbek:
                    query = GoogleTranslator(source='auto', target='en').translate(prompt)

                # Get AI Response
                response = chain.invoke(query)

                # Translate Response
                if is_uzbek:
                    response = GoogleTranslator(source='en', target='uz').translate(response)

                st.markdown(response)
                
                # Save to History
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error: {e}")