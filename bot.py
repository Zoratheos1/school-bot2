import streamlit as st
import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="School AI Genius",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style for the sidebar buttons to look like chat history */
    div.stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 8px;
        margin-bottom: 5px;
    }
    
    /* Style for recommendation chips */
    .rec-button {
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
        background-color: white;
        transition: 0.3s;
    }
    .rec-button:hover {
        background-color: #f0f2f6;
        border-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
PDF_FILE_NAME = "School_constitution.pdf"

# --- SESSION STATE MANAGEMENT ---
if "history" not in st.session_state:
    # This stores ALL chats: { "session_id": [messages], ... }
    st.session_state.history = {}

if "current_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.history[new_id] = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = ["Uniform rules", "Attendance policy", "Grading system"]

# --- THE BRAIN (Cached) ---
@st.cache_resource
def setup_brain():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        
        if not os.path.exists(PDF_FILE_NAME):
            st.error("Error: PDF not found.")
            return None, None

        loader = PyPDFLoader(PDF_FILE_NAME)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        
        llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=api_key)
        
        return retriever, llm
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None

# --- SMART FUNCTIONS ---
def interpret_question(llm, user_input):
    """Detects language and intent to fix logic errors."""
    sys_prompt = """
    You are a Query Interpreter. 
    1. Translate mixed/Uzbek text into a perfect English search query for a school rulebook.
    2. Identify the language the user is speaking (Uzbek or English).
    Output format: Language | English Search Query
    """
    try:
        response = llm.invoke(f"{sys_prompt}\nInput: {user_input}").content.strip()
        if "|" in response:
            return response.split("|", 1)
        return "English", user_input
    except:
        return "English", user_input

def get_answer(llm, retriever, english_query, target_lang):
    """Searches PDF and answers in target language."""
    docs = retriever.invoke(english_query)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""
    Context: {context_text}
    Question: {english_query}
    Instructions: Answer strictly based on context. If unknown, say 'I don't know'.
    Respond in: {target_lang}.
    """
    return llm.invoke(prompt).content

def get_suggestions(llm, topic, lang):
    """Generates 3 follow-up questions."""
    try:
        prompt = f"Based on '{topic}', list 3 short questions a student might ask next. In {lang}. Format: Q1|Q2|Q3."
        return llm.invoke(prompt).content.split("|")[:3]
    except:
        return []

# --- SIDEBAR: CHAT HISTORY ---
with st.sidebar:
    st.title("🎓 History")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.history[new_id] = []
        st.session_state.recommendations = ["Uniform rules", "Attendance policy", "Grading system"]
        st.rerun()
    
    st.markdown("---")
    st.write("**Previous Chats:**")
    
    # 2. List Old Chats
    # We loop through saved sessions and make a button for each
    session_ids = list(st.session_state.history.keys())
    # Reverse to show newest first
    for sess_id in reversed(session_ids):
        # Determine a name for the chat (use first message or "Empty Chat")
        msgs = st.session_state.history[sess_id]
        if msgs:
            # Taking the first 20 chars of the first user message
            chat_name = msgs[0]['content'][:25] + "..."
        else:
            chat_name = "New Conversation"
            
        # Highlight the active chat
        if sess_id == st.session_state.current_session_id:
            if st.button(f" {chat_name}", key=sess_id):
                pass # Already active
        else:
            if st.button(f" {chat_name}", key=sess_id):
                st.session_state.current_session_id = sess_id
                st.rerun()

# --- MAIN CHAT AREA ---
retriever, llm = setup_brain()

# Get messages for the CURRENT active session
current_messages = st.session_state.history[st.session_state.current_session_id]

# 1. Display Chat Messages
st.title("🎓 School AI ")

if not current_messages:
    st.info("👋 Hello! Ask me anything about the school rules. I speak Uzbek & English.")

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Recommendation Buttons (Vertical Stack Fix)
def set_input(text):
    st.session_state.user_input = text

if st.session_state.recommendations:
    st.write("### 💡 Suggested Questions:")
    for rec in st.session_state.recommendations:
        if st.button(rec.strip(), key=rec + st.session_state.current_session_id):
            set_input(rec)
            st.rerun()

# 3. Handle Input
user_input = st.chat_input("Type your question here...")

if "user_input" in st.session_state and st.session_state.user_input:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input and llm:
    # Save User Message to History
    st.session_state.history[st.session_state.current_session_id].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Logic Fix + Translation
                target_lang, search_query = interpret_question(llm, user_input)
                
                # Answer
                response_text = get_answer(llm, retriever, search_query, target_lang)
                st.markdown(response_text)
                
                # Save Assistant Message to History
                st.session_state.history[st.session_state.current_session_id].append({"role": "assistant", "content": response_text})
                
                # New Recommendations
                new_recs = get_suggestions(llm, search_query, target_lang)
                if new_recs:
                    st.session_state.recommendations = new_recs
                    st.rerun()
            except Exception as e:
                st.error("I'm having trouble thinking right now. Please try again.")
