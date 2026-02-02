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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="School AI Genius",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
PDF_FILE_NAME = "School_constitution.pdf"

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = ["Uniform rules", "Attendance policy", "Grading system"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 School Genius")
    st.markdown("I understand **English**, **Uzbek**, and **Mixed** languages!")
    
    if st.button("🔄 Start New Chat", type="primary"):
        st.session_state.messages = []
        st.session_state.recommendations = ["Uniform rules", "Attendance policy", "Grading system"]
        st.rerun()

# --- THE BRAIN ---
@st.cache_resource
def setup_brain():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        
        # 1. Load PDF
        if not os.path.exists(PDF_FILE_NAME):
            st.error("Error: PDF not found.")
            return None, None

        loader = PyPDFLoader(PDF_FILE_NAME)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        # 2. Embeddings (Memory)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        
        # 3. LLM (Llama 3.3 - The Smartest Available)
        llm = ChatGroq(
            temperature=0.0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=api_key
        )
        
        return retriever, llm
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None

# --- INTELLIGENT FUNCTIONS ---

def interpret_question(llm, user_input):
    """
    Step 1: Pattern Recognition.
    Reads mixed language (Uzbek/English) and outputs a CLEAN English search query.
    Also detects the user's target language.
    """
    sys_prompt = """
    You are a Query Interpreter. Analyze the user's input.
    1. Identify the user's intent clearly.
    2. Translate mixed/Uzbek text into a perfect English search query for a school rulebook.
    3. Identify the language the user is speaking (Uzbek or English).
    
    Output format: Language | English Search Query
    
    Example 1:
    Input: "Mening uniformam qanaqa bo'lishi kerak?"
    Output: Uzbek | What are the school uniform and dress code requirements?

    Example 2:
    Input: "can I wear sport kiyim?"
    Output: Uzbek | Can students wear sports clothes or tracksuits in school?
    
    Example 3:
    Input: "What is the grading scale?"
    Output: English | What is the grading system and scale?
    """
    
    try:
        response = llm.invoke(f"{sys_prompt}\nInput: {user_input}")
        content = response.content.strip()
        
        if "|" in content:
            lang, query = content.split("|", 1)
            return lang.strip(), query.strip()
        else:
            return "English", user_input
    except:
        return "English", user_input

def get_answer(llm, retriever, english_query, target_lang):
    """
    Step 2 & 3: Search and Answer in the correct language.
    """
    # Search PDF using the CLEAN English query
    docs = retriever.invoke(english_query)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Generate Answer
    prompt = f"""
    You are a School Administrator. Use the following context to answer the question.
    
    Context:
    {context_text}
    
    Question: {english_query}
    
    Instructions:
    1. Answer strictly based on the context.
    2. If the answer is not in the context, say "I don't know" (or Uzbek equivalent).
    3. The user speaks: {target_lang}. WRITE YOUR ANSWER IN {target_lang}.
    4. Be professional and clear.
    """
    
    response = llm.invoke(prompt)
    return response.content

def get_suggestions(llm, topic, lang):
    """Generates follow-up questions in the USER'S language."""
    try:
        prompt = f"Based on the topic '{topic}', list 3 short follow-up questions a student might ask. Write them in {lang}. Format: Q1|Q2|Q3. No numbers."
        response = llm.invoke(prompt).content
        questions = response.split("|")
        return [q.strip() for q in questions[:3]]
    except:
        return []

# --- MAIN APP LOGIC ---

retriever, llm = setup_brain()

# 1. Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Recommendation Buttons
def set_input(text):
    st.session_state.user_input = text

col1, col2, col3 = st.columns(3)
if len(st.session_state.recommendations) >= 3:
    if col1.button(st.session_state.recommendations[0]): set_input(st.session_state.recommendations[0])
    if col2.button(st.session_state.recommendations[1]): set_input(st.session_state.recommendations[1])
    if col3.button(st.session_state.recommendations[2]): set_input(st.session_state.recommendations[2])

# 3. Input Handling
user_input = st.chat_input("Ask about school rules (Uzbek or English)...")

if "user_input" in st.session_state and st.session_state.user_input:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input and llm:
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # A. INTERPRET (Fixes the "cloth vs subject" logic error)
                target_lang, search_query = interpret_question(llm, user_input)
                # (Optional: Print debug to console to see the translation working)
                print(f"User Lang: {target_lang} | Search Query: {search_query}")

                # B. SEARCH & ANSWER
                response_text = get_answer(llm, retriever, search_query, target_lang)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # C. NEW SUGGESTIONS
                new_recs = get_suggestions(llm, search_query, target_lang)
                if new_recs:
                    st.session_state.recommendations = new_recs
                    st.rerun()
                    
            except Exception as e:
                st.error("I'm having trouble thinking right now. Please try again.")
