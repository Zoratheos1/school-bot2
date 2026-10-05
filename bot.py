import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------
# 1. Page Configuration & Custom CSS
# --------------------------------------------
st.set_page_config(page_title="School Assistant", page_icon="🎓", layout="wide")

custom_css = """
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Typography & Accessibility */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 18px !important; 
    }
    /* Primary Buttons & Accents */
    .stButton>button {
        background-color: #0E1117;
        color: #39FF14;
        border: 2px solid #39FF14;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #39FF14;
        color: #0E1117;
    }
    /* Chat Bubbles */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: transparent;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    /* User Chat Bubble */
    .stChatMessage:nth-child(odd) {
        background-color: #1E1E1E;
        border: none;
    }
    /* AI Chat Bubble */
    .stChatMessage:nth-child(even) {
        border: 1px solid #39FF14;
        background-color: rgba(57, 255, 20, 0.05);
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --------------------------------------------
# 2. Application Logic & Functions
# --------------------------------------------
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for the Vector Store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates an in-memory FAISS vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """Sets up the Groq Llama-3 chain with a strict zero-hallucination prompt."""
    llm = ChatGroq(
        groq_api_key=st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=0.1
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are 'School Assistant', a highly accurate, professional, and accessible AI guide. Your primary users include students, parents, and elderly citizens. You must explain things in clear, simple language. You must answer questions based ONLY on the provided context retrieved from the uploaded documents. Do not use outside knowledge. If the answer is not contained in the text, you must explicitly say: 'I cannot find this information in the official documents provided.' Always cite the document or rule number if available in the text. Communicate fluently in the language the user types (Uzbek, Russian, or English).\n\nContext: {context}"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --------------------------------------------
# 3. Sidebar Configuration
# --------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ School Assistant")
    # You can add your geometric logo here:
    # st.image("logo.png", use_container_width=True)
    
    st.markdown("---")
    st.markdown("📂 **Upload Documents**")
    pdf_docs = st.file_uploader("Upload Rulebooks or Constitutions (PDF)", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("🟢 AI Ready")
        else:
            st.warning("Please upload a PDF first.")
            
    st.markdown("---")
    if "vector_store" not in st.session_state:
        st.error("🔴 Database Empty")
    else:
        st.success("🟢 AI Ready")
        
    with st.expander("⚙️ Tech Stack"):
        st.markdown("- **Frontend:** Streamlit\n- **AI Model:** Groq (Llama-3)\n- **Memory:** FAISS Vector Store\n- **Logic:** LangChain & Python")

# --------------------------------------------
# 4. Main Chat Interface
# --------------------------------------------
st.title("Welcome to School Assistant")
st.markdown("Ask me anything about school rules, schedules, or uploaded constitutions.")

# Quick suggestion buttons
col1, col2, col3 = st.columns(3)
if col1.button("Uniform Policy?"):
    st.session_state.quick_ask = "What is the uniform policy?"
if col2.button("Grading Rules?"):
    st.session_state.quick_ask = "How are grades calculated?"
if col3.button("Citizen Rights?"):
    st.session_state.quick_ask = "What are the core citizen rights mentioned?"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle quick suggestions or standard chat input
user_question = st.chat_input("Ask a question...")
if "quick_ask" in st.session_state and st.session_state.quick_ask:
    user_question = st.session_state.quick_ask
    st.session_state.quick_ask = None

if user_question:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate AI response
    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_conversational_chain()
                response = chain.invoke({"input": user_question})
                ai_answer = response["answer"]
                st.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload and process a document in the sidebar first.")
