# app.py - Enhanced Dark Mode with CodeToLive Styling
import streamlit as st
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
import tempfile
import shutil
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load_dotenv()

# Add current directory to path
sys.path.append('.')

# Try to import RAG components
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from langchain_groq import ChatGroq
    import numpy as np
    import uuid
    
    # Import document processing modules
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Check API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not found in .env file")
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.info("Install: pip install sentence-transformers chromadb langchain-groq pypdf langchain-community langchain-text-splitters")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark mode
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant - CodeToLive",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS for both light and dark modes with better Streamlit component support
def get_css(dark_mode):
    theme = "dark-mode" if dark_mode else ""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Theme Variables */
    :root {{
        --primary-color: #1abc9c;
        --primary-dark: #16a085;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --dark-color: #2c3e50;
        --light-color: #f4f7f6;
        --bg-color: #ffffff;
        --card-bg: #ffffff;
        --border-color: #e0e0e0;
        --text-color: #333333;
        --text-light: #666666;
        --shadow-light: 0 5px 15px rgba(0, 0, 0, 0.05);
        --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.1);
        --radius: 10px;
    }}
    
    .dark-mode {{
        --primary-color: #1abc9c;
        --primary-dark: #16a085;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --dark-color: #ffffff;
        --light-color: #1a1a1a;
        --bg-color: #121212;
        --card-bg: #1e1e1e;
        --border-color: #2d2d2d;
        --text-color: #ffffff;
        --text-light: #a0a0a0;
        --shadow-light: 0 5px 15px rgba(0, 0, 0, 0.3);
        --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.4);
    }}
    
    /* Apply theme to body */
    body {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        transition: all 0.3s ease;
    }}
    
    /* Main Container */
    .stApp {{
        max-width: 1400px;
        margin: 0 auto;
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}
    
    /* Fix for all text elements */
    .stMarkdown, .stText, .stAlert, .stSuccess, .stWarning, .stError, .stInfo,
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: var(--text-color) !important;
    }}
    
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        padding: 25px 30px;
        border-radius: var(--radius);
        margin-bottom: 30px;
        box-shadow: var(--shadow-medium);
        text-align: center;
    }}
    
    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: white !important;
    }}
    
    .main-header p {{
        font-size: 1.1rem;
        opacity: 0.95;
        max-width: 800px;
        margin: 0 auto;
        color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Card Styling */
    .card {{
        background-color: var(--card-bg);
        border-radius: var(--radius);
        padding: 25px;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: var(--shadow-medium);
    }}
    
    .card-header {{
        background: transparent !important;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 15px;
        margin-bottom: 20px;
        font-weight: 600;
        font-size: 1.3rem;
    }}
    
    /* Upload Area */
    .upload-area {{
        border: 3px dashed var(--primary-color);
        border-radius: var(--radius);
        padding: 50px 20px;
        text-align: center;
        background: { 'rgba(26, 188, 156, 0.05)' if not dark_mode else 'rgba(26, 188, 156, 0.1)' };
        margin: 20px 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .upload-area:hover {{
        background: { 'rgba(26, 188, 156, 0.1)' if not dark_mode else 'rgba(26, 188, 156, 0.15)' };
        border-color: var(--primary-dark);
        transform: translateY(-2px);
    }}
    
    /* Chat Messages */
    .user-msg {{
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        padding: 15px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: var(--shadow-light);
        border: none;
    }}
    
    .assistant-msg {{
        background-color: var(--card-bg);
        padding: 15px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px 0;
        max-width: 70%;
        margin-right: auto;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }}
    
    /* Streamlit Component Overrides */
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white !important;
        border: none !important;
        padding: 10px 25px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(26, 188, 156, 0.3);
    }}
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {{
        background: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stButton > button[kind="secondary"]:hover {{
        background: var(--light-color) !important;
    }}
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--primary-color) !important;
    }}
    
    /* Select boxes */
    [data-baseweb="select"] > div {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    [data-baseweb="select"] input {{
        color: var(--text-color) !important;
    }}
    
    /* Sliders */
    .stSlider > div > div {{
        color: var(--primary-color) !important;
    }}
    
    .stSlider > div > div > div {{
        background-color: var(--primary-color) !important;
    }}
    
    /* Checkboxes and Radio buttons */
    .stCheckbox, .stRadio {{
        color: var(--text-color) !important;
    }}
    
    .stCheckbox > label, .stRadio > label {{
        color: var(--text-color) !important;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-color) !important;
    }}
    
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: var(--light-color) !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0 0 8px 8px !important;
        margin-top: 0 !important;
    }}
    
    /* Chat input */
    .stChatInput {{
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        background-color: var(--card-bg) !important;
    }}
    
    .stChatInput input {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        border: 2px dashed var(--border-color) !important;
        background-color: var(--card-bg) !important;
        border-radius: var(--radius) !important;
    }}
    
    .stFileUploader > div:hover {{
        border-color: var(--primary-color) !important;
    }}
    
    /* Status boxes */
    .success-box {{
        background: { 'linear-gradient(135deg, #d4edda, #c3e6cb)' if not dark_mode else 'linear-gradient(135deg, #1e4620, #2d5a2f)' } !important;
        color: { '#155724' if not dark_mode else '#c8e6c9' } !important;
        padding: 20px;
        border-radius: var(--radius);
        border: 2px solid { '#c3e6cb' if not dark_mode else '#2d5a2f' } !important;
        margin: 15px 0;
        font-weight: 500;
    }}
    
    .info-box {{
        background: { 'linear-gradient(135deg, #d1ecf1, #bee5eb)' if not dark_mode else 'linear-gradient(135deg, #1a3c4e, #2c3e50)' } !important;
        color: { '#0c5460' if not dark_mode else '#b3e0f2' } !important;
        padding: 20px;
        border-radius: var(--radius);
        border: 2px solid { '#bee5eb' if not dark_mode else '#2c3e50' } !important;
        margin: 15px 0;
        font-weight: 500;
    }}
    
    .warning-box {{
        background: { 'linear-gradient(135deg, #fff3cd, #ffeaa7)' if not dark_mode else 'linear-gradient(135deg, #5d4037, #795548)' } !important;
        color: { '#856404' if not dark_mode else '#ffccbc' } !important;
        padding: 20px;
        border-radius: var(--radius);
        border: 2px solid { '#ffeaa7' if not dark_mode else '#795548' } !important;
        margin: 15px 0;
        font-weight: 500;
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: var(--radius);
        text-align: center;
        box-shadow: var(--shadow-light);
        border-top: 4px solid var(--primary-color);
    }}
    
    .metric-card h3 {{
        color: var(--text-color) !important;
        margin: 0;
    }}
    
    .metric-card p {{
        color: var(--text-light) !important;
        margin: 5px 0 0 0;
    }}
    
    /* Badges */
    .badge {{
        background: var(--primary-color);
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 2px;
    }}
    
    /* File cards */
    .file-card {{
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: var(--radius);
        margin: 15px 0;
        border-left: 5px solid var(--primary-color);
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }}
    
    .file-card:hover {{
        transform: translateX(5px);
    }}
    
    /* Divider */
    .divider {{
        height: 2px;
        background: linear-gradient(to right, transparent, var(--primary-color), transparent);
        margin: 30px 0;
    }}
    
    /* Feature icons */
    .feature-icon {{
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px;
        color: white !important;
        font-size: 24px;
    }}
    
    /* Toggle switch for dark mode */
    .toggle-switch {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        background: var(--card-bg);
        border-radius: var(--radius);
        border: 1px solid var(--border-color);
        margin-bottom: 20px;
    }}
    
    /* Ensure all text in Streamlit widgets is properly colored */
    div[data-testid="stMetricValue"], 
    div[data-testid="stMetricLabel"],
    .st-bb, .st-bc, .st-bd, .st-be, .st-bf, .st-bg, .st-bh, .st-bi, .st-bj, .st-bk, .st-bl, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, .st-bw, .st-bx, .st-by, .st-bz {{
        color: var(--text-color) !important;
    }}
    
    /* Fix for streamlit's default alerts */
    .stAlert {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    </style>
    """

# Apply CSS based on current mode
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Add theme class to body for CSS variables
if st.session_state.dark_mode:
    st.markdown('<div class="dark-mode">', unsafe_allow_html=True)

# Initialize RAG System
def initialize_system():
    """Initialize the RAG system"""
    try:
        with st.spinner("🔄 Initializing system components..."):
            # Initialize embedding model
            st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector store
            os.makedirs("./data/vector_store", exist_ok=True)
            client = chromadb.PersistentClient(path="./data/vector_store")
            st.session_state.vectorstore = client.get_or_create_collection(
                name="pdf_documents",
                metadata={"description": "PDF document embeddings"}
            )
            
            # Initialize LLM
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found")
            
            st.session_state.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1024
            )
            
            st.session_state.system_initialized = True
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return False

# Process uploaded PDFs
def process_uploaded_pdfs(uploaded_files, append_mode=True):
    """Process uploaded PDF files and add to vector database"""
    try:
        if not uploaded_files:
            return {"success": False, "message": "No files uploaded"}
        
        # Create temp directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        
        all_chunks = []
        processed_files = []
        
        # Save and process each uploaded file
        for uploaded_file in uploaded_files:
            # Save file
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata['source_file'] = uploaded_file.name
                    doc.metadata['uploaded'] = True
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                
                processed_files.append({
                    'name': uploaded_file.name,
                    'pages': len(documents),
                    'chunks': len(chunks)
                })
                
            except Exception as e:
                st.warning(f"Could not process {uploaded_file.name}: {str(e)}")
                continue
        
        if not all_chunks:
            return {"success": False, "message": "No valid PDF content found"}
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in all_chunks]
        embeddings = st.session_state.embedding_model.encode(texts)
        
        # Prepare data for ChromaDB
        ids = []
        documents_text = []
        metadatas = []
        
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            # Generate unique ID
            doc_id = f"upload_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            documents_text.append(chunk.page_content)
            
            # Prepare metadata
            metadata = dict(chunk.metadata)
            metadata['chunk_index'] = i
            metadata['content_length'] = len(chunk.page_content)
            metadata['upload_timestamp'] = str(uuid.uuid4())
            metadatas.append(metadata)
        
        # Add to vector store
        st.session_state.vectorstore.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents_text,
            metadatas=metadatas
        )
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        
        return {
            "success": True,
            "message": f"✅ Successfully processed {len(processed_files)} file(s)",
            "files": processed_files,
            "total_chunks": len(all_chunks)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Processing error: {str(e)}"}

# Clear vector database
def clear_vector_database():
    """Clear all documents from vector database"""
    try:
        if st.session_state.vectorstore:
            # Create new collection to clear data
            client = chromadb.PersistentClient(path="./data/vector_store")
            client.delete_collection("pdf_documents")
            
            # Recreate collection
            st.session_state.vectorstore = client.create_collection(
                name="pdf_documents",
                metadata={"description": "PDF document embeddings"}
            )
            
            return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
    return False

# RAG Query Function
def query_rag_system(question: str, top_k: int = 3):
    """Query the RAG system"""
    try:
        # Generate query embedding
        query_embedding = st.session_state.embedding_model.encode([question])[0]
        
        # Search in vector store
        results = st.session_state.vectorstore.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        sources = []
        context = ""
        
        if results['documents'] and len(results['documents'][0]) > 0:
            retrieved_docs = results['documents'][0]
            retrieved_metadata = results['metadatas'][0]
            retrieved_distances = results['distances'][0]
            
            for i, (doc, metadata, distance) in enumerate(zip(
                retrieved_docs, retrieved_metadata, retrieved_distances
            )):
                similarity_score = 1 - distance
                sources.append({
                    'content': doc[:500] + "..." if len(doc) > 500 else doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score
                })
            
            context = "\n\n".join(retrieved_docs)
        else:
            context = "No relevant documents found."
        
        # Generate answer
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the context provided
2. If context doesn't contain enough information, say so
3. Be concise and accurate
4. Don't make up information

ANSWER:"""
        
        response = st.session_state.llm.invoke([prompt])
        answer = response.content
        
        return {
            'answer': answer,
            'sources': sources,
            'context_found': len(sources) > 0
        }
        
    except Exception as e:
        return {
            'answer': f"❌ Error: {str(e)}",
            'sources': [],
            'context_found': False
        }

# Toggle dark mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Main App
def main():
    # Apply theme wrapper
    theme_class = "dark-mode" if st.session_state.dark_mode else ""
    
    # Main Header
    st.markdown(f"""
    <div class="main-header">
        <h1>📚 Document Q&A Assistant</h1>
        <p>Upload PDFs and ask questions about them in real-time!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: #1abc9c; margin-bottom: 5px;">CodeToLive</h2>
            <p style="color: {'#a0a0a0' if st.session_state.dark_mode else '#666'}; font-size: 0.9rem;">
                Document Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dark Mode Toggle
        st.markdown("### 🌓 Theme")
        col1, col2 = st.columns([1, 1])
        with col1:
            dark_mode_label = "🌙 Dark" if st.session_state.dark_mode else "☀️ Light"
            if st.button(dark_mode_label, use_container_width=True, type="primary" if st.session_state.dark_mode else "secondary"):
                toggle_dark_mode()
                st.rerun()
        
        with col2:
            if st.button("⚙️ Settings", use_container_width=True, type="secondary"):
                st.session_state.show_settings = not st.session_state.show_settings
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown("### 🔧 System Control")
        
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                if initialize_system():
                    st.success("✅ System initialized!")
                    st.rerun()
        else:
            st.success("✅ System Ready")
            
            # Database Stats
            st.markdown("### 📊 Database Stats")
            try:
                doc_count = st.session_state.vectorstore.count()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{doc_count}</h3>
                    <p>Documents</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>0</h3>
                    <p>Documents</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Settings (only show if toggled)
            if st.session_state.show_settings:
                st.markdown("### ⚙️ Settings")
                top_k = st.slider("Reference chunks", 1, 10, 3, help="Number of document chunks to reference")
                show_sources = st.checkbox("Show sources", value=True)
            else:
                top_k = 3
                show_sources = True
            
            # Clear buttons
            st.markdown("### 🗑️ Data Management")
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Clear Database", use_container_width=True, type="secondary"):
                    if clear_vector_database():
                        st.success("Database cleared!")
                        st.rerun()
            
            with col4:
                if st.button("Clear Chat", use_container_width=True, type="secondary"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("📋 Summarize All", use_container_width=True, type="secondary"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "Provide a comprehensive summary of all uploaded documents"
            })
            st.rerun()
        
        if st.button("🔍 Key Topics", use_container_width=True, type="secondary"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "What are the main topics covered in these documents?"
            })
            st.rerun()
        
        if st.button("💡 Ask Sample", use_container_width=True, type="secondary"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "What can you tell me about these documents?"
            })
            st.rerun()
        
        # Footer
        st.markdown(f"""
        <div style="text-align: center; margin-top: 30px; color: {'#a0a0a0' if st.session_state.dark_mode else '#666'};">
            <p>Powered by CodeToLive</p>
            <p style="font-size: 0.8rem;">RAG Technology</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📤 Upload Documents</div>', unsafe_allow_html=True)
        
        # Upload area
        st.markdown(f"""
        <div class="upload-area">
            <div class="feature-icon" style="background: transparent; color: var(--primary-color); font-size: 40px;">
                📤
            </div>
            <h3>Drag & Drop PDF Files</h3>
            <p>or click to browse your files</p>
            <p style="color: var(--text-light); margin-top: 10px;">
                Limit 200MB per file • PDF only
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            # Show uploaded files
            st.markdown("### 📄 Uploaded Files")
            for uploaded_file in uploaded_files:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.markdown(f"""
                <div class="file-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{uploaded_file.name}</strong><br>
                            <small style="color: var(--text-light);">Size: {file_size_mb:.2f} MB</small>
                        </div>
                        <span class="badge">PDF</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Process button
            if st.button("⚡ Process & Add to Database", type="primary", use_container_width=True):
                if not st.session_state.system_initialized:
                    st.error("⚠️ Please initialize system first")
                elif uploaded_files:
                    with st.spinner("🔄 Processing documents..."):
                        result = process_uploaded_pdfs(uploaded_files)
                        
                        if result["success"]:
                            st.markdown(f'<div class="success-box">{result["message"]}</div>', unsafe_allow_html=True)
                            
                            # Show processing details
                            for file_info in result.get("files", []):
                                st.markdown(f"""
                                <div style="background: var(--light-color); padding: 10px; border-radius: 5px; margin: 5px 0;">
                                    ✅ <strong>{file_info['name']}</strong><br>
                                    <small style="color: var(--text-light);">
                                        Pages: {file_info['pages']} • Chunks: {file_info['chunks']}
                                    </small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.balloons()
                            st.success(f"✨ Total chunks added: {result['total_chunks']}")
                            st.rerun()
                        else:
                            st.markdown(f'<div class="warning-box">{result["message"]}</div>', unsafe_allow_html=True)
        
        # How it works
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">💡 How it works:</h4>
            <ol style="margin-bottom: 0;">
                <li>Upload PDF files</li>
                <li>Click "Process & Add to Database"</li>
                <li>Ask questions about the documents</li>
                <li>Get AI-powered answers with sources</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">💬 Ask Questions</div>', unsafe_allow_html=True)
        
        if not st.session_state.system_initialized:
            st.markdown(f"""
            <div class="warning-box" style="text-align: center;">
                <h4>👈 Please Initialize System</h4>
                <p>Use the button in the sidebar to initialize the system first</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Chat container
            chat_container = st.container(height=400)
            
            with chat_container:
                # Display chat history
                if not st.session_state.chat_history:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 40px 20px;">
                        <div class="feature-icon" style="margin: 0 auto 20px;">
                            🤖
                        </div>
                        <h3>Hello! I'm your Document Assistant</h3>
                        <p style="color: var(--text-light);">
                            Upload a PDF document and ask me anything about its content.
                        </p>
                        <div style="margin-top: 20px;">
                            <span class="badge">Summarize</span>
                            <span class="badge">Find Information</span>
                            <span class="badge">Extract Key Points</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            st.markdown(f'<div class="user-msg"><strong>You:</strong> {message["content"]}</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="assistant-msg"><strong>🤖 Assistant:</strong> {message["content"]}</div>', 
                                      unsafe_allow_html=True)
                            
                            # Show sources
                            if show_sources and message.get("sources"):
                                with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                                    for i, source in enumerate(message["sources"]):
                                        source_name = source.get('metadata', {}).get('source_file', 'Unknown Document')
                                        score = source.get('similarity_score', 0)
                                        
                                        # Create source card
                                        st.markdown(f"""
                                        <div style="background: var(--light-color); padding: 15px; border-radius: 8px; margin: 10px 0;">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <strong>{source_name}</strong>
                                                <span class="badge">{score:.1%} relevant</span>
                                            </div>
                                            <p style="margin: 10px 0 0 0; font-style: italic; color: var(--text-light);">
                                                "{source['content']}"
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
            
            # Chat input
            if st.session_state.system_initialized:
                question = st.chat_input("Ask a question about your documents...")
                
                if question:
                    # Add user message
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    # Show thinking
                    with st.spinner("🔍 Searching documents..."):
                        # Get response
                        response = query_rag_system(question, top_k)
                        
                        # Add assistant response
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "sources": response['sources']
                        })
                    
                    # Rerun to show new messages
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
        
        # Quick Questions
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚡ Quick Questions</div>', unsafe_allow_html=True)
        
        qcol1, qcol2 = st.columns(2)
        
        with qcol1:
            if st.button("📋 Summarize", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "Can you summarize the main points of all documents?"
                })
                st.rerun()
            
            if st.button("🔑 Key Points", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "What are the key points I should remember?"
                })
                st.rerun()
        
        with qcol2:
            if st.button("📖 Main Topics", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "List the main topics covered in these documents"
                })
                st.rerun()
            
            if st.button("❓ Ask Sample", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "What is this document about?"
                })
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    # Features Grid
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### ✨ Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div class="feature-icon">
                📄
            </div>
            <h4>PDF Support</h4>
            <p style="color: var(--text-light);">
                Upload and analyze any PDF document with high accuracy text extraction.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div class="feature-icon">
                🧠
            </div>
            <h4>AI-Powered</h4>
            <p style="color: var(--text-light);">
                Advanced RAG system with semantic search and intelligent question answering.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div class="feature-icon">
                🔒
            </div>
            <h4>Secure & Private</h4>
            <p style="color: var(--text-light);">
                Your documents are processed locally. No data is shared with third parties.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 2])
    
    with footer_col2:
        st.markdown(f"""
        <div style="text-align: center; color: var(--text-light); padding: 20px;">
            <p style="font-size: 0.9rem;">Made with ❤️ by</p>
            <h3 style="color: #1abc9c; margin: 0;">CodeToLive</h3>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
