# # app.py - Working Q&A App
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from typing import List, Dict, Any
# import sys
# import warnings

# # Suppress warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load_dotenv()

# # Add the path to your RAG modules
# sys.path.append('.')

# # Import your RAG components
# try:
#     from sentence_transformers import SentenceTransformer
#     import chromadb
#     from langchain_groq import ChatGroq
#     import numpy as np
    
#     # Check if GROQ API key is available
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     if not GROQ_API_KEY:
#         st.error("⚠️ GROQ_API_KEY not found in .env file")
#         st.info("Please add your GROQ API key to the .env file")
# except ImportError as e:
#     st.error(f"Missing dependencies: {e}")
#     st.info("Run: pip install sentence-transformers chromadb langchain-groq")

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'vectorstore_initialized' not in st.session_state:
#     st.session_state.vectorstore_initialized = False
# if 'embedding_model' not in st.session_state:
#     st.session_state.embedding_model = None
# if 'vectorstore' not in st.session_state:
#     st.session_state.vectorstore = None
# if 'llm' not in st.session_state:
#     st.session_state.llm = None

# # Page configuration
# st.set_page_config(
#     page_title="Document Q&A Assistant",
#     page_icon="🤖",
#     layout="centered"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .stApp {
#         max-width: 900px;
#         margin: 0 auto;
#     }
#     .chat-container {
#         padding: 20px;
#     }
#     .user-msg {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 12px 16px;
#         border-radius: 18px 18px 4px 18px;
#         margin: 8px 0;
#         max-width: 80%;
#         margin-left: auto;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }
#     .assistant-msg {
#         background: #f0f2f6;
#         color: #1a1a1a;
#         padding: 12px 16px;
#         border-radius: 18px 18px 18px 4px;
#         margin: 8px 0;
#         max-width: 80%;
#         margin-right: auto;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }
#     .thinking {
#         display: flex;
#         align-items: center;
#         gap: 10px;
#         padding: 15px;
#         background: #f8f9fa;
#         border-radius: 10px;
#         margin: 10px 0;
#         animation: pulse 1.5s infinite;
#     }
#     @keyframes pulse {
#         0% { opacity: 0.6; }
#         50% { opacity: 1; }
#         100% { opacity: 0.6; }
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize RAG System
# def initialize_rag_system():
#     """Initialize the RAG system components"""
#     try:
#         with st.spinner("🔄 Initializing system..."):
#             # 1. Initialize embedding model
#             st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
#             # 2. Initialize vector store
#             client = chromadb.PersistentClient(path="./data/vector_store")
#             st.session_state.vectorstore = client.get_or_create_collection(
#                 name="pdf_documents",
#                 metadata={"description": "PDF document embeddings"}
#             )
            
#             # 3. Initialize LLM
#             GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#             if not GROQ_API_KEY:
#                 raise ValueError("GROQ_API_KEY not found in .env file")
            
#             st.session_state.llm = ChatGroq(
#                 groq_api_key=GROQ_API_KEY,
#                 model_name="llama-3.1-8b-instant",
#                 temperature=0.1,
#                 max_tokens=1024
#             )
            
#             # Check if vector store has data
#             doc_count = st.session_state.vectorstore.count()
#             if doc_count == 0:
#                 st.warning("⚠️ Vector store is empty. Please process documents first.")
#                 st.info("""
#                 To process documents:
#                 1. Place PDFs in `data/pdfs/` folder
#                 2. Run the processing script:
#                    ```bash
#                    python process_documents.py
#                    ```
#                 """)
            
#             st.session_state.vectorstore_initialized = True
#             return True
            
#     except Exception as e:
#         st.error(f"❌ Failed to initialize system: {str(e)}")
#         return False

# # RAG Query Function
# def query_rag_system(question: str, top_k: int = 3):
#     """Query the RAG system with a question"""
#     try:
#         # Generate query embedding
#         query_embedding = st.session_state.embedding_model.encode([question])[0]
        
#         # Search in vector store
#         results = st.session_state.vectorstore.query(
#             query_embeddings=[query_embedding.tolist()],
#             n_results=top_k
#         )
        
#         sources = []
#         context = ""
        
#         if results['documents'] and len(results['documents'][0]) > 0:
#             # Prepare context from retrieved documents
#             retrieved_docs = results['documents'][0]
#             retrieved_metadata = results['metadatas'][0]
#             retrieved_distances = results['distances'][0]
            
#             for i, (doc, metadata, distance) in enumerate(zip(
#                 retrieved_docs, retrieved_metadata, retrieved_distances
#             )):
#                 similarity_score = 1 - distance
#                 sources.append({
#                     'content': doc[:500] + "..." if len(doc) > 500 else doc,
#                     'metadata': metadata,
#                     'similarity_score': similarity_score
#                 })
            
#             # Combine context
#             context = "\n\n".join(retrieved_docs)
#         else:
#             context = "No relevant documents found in the database."
        
#         # Prepare prompt for LLM
#         prompt = f"""You are a helpful AI assistant. Use the following context to answer the question accurately.

# CONTEXT FROM DOCUMENTS:
# {context}

# QUESTION: {question}

# INSTRUCTIONS:
# 1. Answer based ONLY on the context provided
# 2. If the context doesn't contain enough information, say: "I don't have enough information to answer this question based on the available documents."
# 3. Be concise and accurate
# 4. Don't make up information

# ANSWER:"""
        
#         # Get response from LLM
#         response = st.session_state.llm.invoke([prompt])
#         answer = response.content
        
#         return {
#             'answer': answer,
#             'sources': sources,
#             'context_found': len(sources) > 0
#         }
        
#     except Exception as e:
#         return {
#             'answer': f"❌ Error processing your question: {str(e)}",
#             'sources': [],
#             'context_found': False
#         }

# # Main App
# def main():
#     st.title("📚 Document Q&A Assistant")
#     st.markdown("Ask questions about your documents and get AI-powered answers.")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # Initialize button
#         if not st.session_state.vectorstore_initialized:
#             if st.button("🚀 Initialize System", use_container_width=True, type="primary"):
#                 if initialize_rag_system():
#                     st.success("✅ System initialized!")
#                     st.rerun()
#         else:
#             st.success("✅ System is ready!")
            
#             # Show document count
#             try:
#                 doc_count = st.session_state.vectorstore.count()
#                 st.metric("Documents in Database", doc_count)
#             except:
#                 st.metric("Documents in Database", 0)
        
#         st.markdown("---")
        
#         # Settings
#         st.subheader("Settings")
#         top_k = st.slider("Number of reference chunks", 1, 10, 3)
        
#         show_sources = st.checkbox("Show source documents", value=True)
        
#         # Clear chat
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.chat_history = []
#             st.rerun()
        
#         st.markdown("---")
#         st.caption("Powered by ChromaDB + Sentence Transformers + Groq LLM")
    
#     # Main chat interface
#     chat_container = st.container()
    
#     with chat_container:
#         # Display chat history
#         for message in st.session_state.chat_history:
#             if message["role"] == "user":
#                 st.markdown(f'<div class="user-msg"><strong>You:</strong> {message["content"]}</div>', 
#                           unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="assistant-msg"><strong>Assistant:</strong> {message["content"]}</div>', 
#                           unsafe_allow_html=True)
                
#                 # Show sources if available
#                 if show_sources and message.get("sources"):
#                     with st.expander(f"📚 Sources ({len(message['sources'])} found)", expanded=False):
#                         for i, source in enumerate(message["sources"]):
#                             source_name = source.get('metadata', {}).get('source_file', 'Unknown document')
#                             score = source.get('similarity_score', 0)
                            
#                             col1, col2 = st.columns([3, 1])
#                             with col1:
#                                 st.markdown(f"**Document:** {source_name}")
#                             with col2:
#                                 st.markdown(f"**Relevance:** {score:.1%}")
                            
#                             st.markdown(f"*{source['content']}*")
#                             st.markdown("---")
    
#     # Chat input
#     if not st.session_state.vectorstore_initialized:
#         st.info("👈 Please initialize the system from the sidebar first.")
#     else:
#         # Quick action buttons
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("📋 Summarize documents", use_container_width=True):
#                 st.session_state.chat_history.append({
#                     "role": "user", 
#                     "content": "Can you provide a summary of all the documents?"
#                 })
#                 st.rerun()
#         with col2:
#             if st.button("🔍 Key topics", use_container_width=True):
#                 st.session_state.chat_history.append({
#                     "role": "user", 
#                     "content": "What are the main topics covered in the documents?"
#                 })
#                 st.rerun()
#         with col3:
#             if st.button("💡 Ask sample", use_container_width=True):
#                 st.session_state.chat_history.append({
#                     "role": "user", 
#                     "content": "What is this document about?"
#                 })
#                 st.rerun()
        
#         st.markdown("---")
        
#         # Chat input
#         question = st.chat_input("Type your question here...")
        
#         if question:
#             # Add user message
#             st.session_state.chat_history.append({"role": "user", "content": question})
            
#             # Show thinking indicator
#             with st.spinner("🤔 Searching documents and generating answer..."):
#                 # Get response from RAG system
#                 response = query_rag_system(question, top_k)
                
#                 # Add assistant response
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": response['answer'],
#                     "sources": response['sources']
#                 })
            
#             # Rerun to show new messages
#             st.rerun()

# # Create processing script
# processing_script = """
# # process_documents.py
# import os
# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import chromadb
# import numpy as np
# import uuid

# def process_documents():
#     print("🚀 Starting document processing...")
    
#     # Create directories if they don't exist
#     os.makedirs("./data/pdfs", exist_ok=True)
#     os.makedirs("./data/vector_store", exist_ok=True)
    
#     # Check for PDFs
#     pdf_dir = Path("./data/pdfs")
#     pdf_files = list(pdf_dir.glob("*.pdf"))
    
#     if not pdf_files:
#         print("❌ No PDF files found in ./data/pdfs/")
#         print("Please add your PDF files to that directory and run again.")
#         return False
    
#     print(f"📄 Found {len(pdf_files)} PDF file(s)")
    
#     # Initialize models
#     print("🔧 Initializing embedding model...")
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     print("💾 Initializing vector database...")
#     client = chromadb.PersistentClient(path="./data/vector_store")
#     collection = client.get_or_create_collection("pdf_documents")
    
#     # Clear existing data (optional)
#     # collection.delete(where={})
    
#     all_documents = []
    
#     # Load and process PDFs
#     for pdf_file in pdf_files:
#         print(f"📖 Processing: {pdf_file.name}")
#         try:
#             loader = PyPDFLoader(str(pdf_file))
#             documents = loader.load()
            
#             # Add metadata
#             for doc in documents:
#                 doc.metadata['source_file'] = pdf_file.name
            
#             all_documents.extend(documents)
#             print(f"   ✅ Loaded {len(documents)} pages")
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
    
#     if not all_documents:
#         print("❌ No documents were loaded")
#         return False
    
#     # Split documents into chunks
#     print("✂️ Splitting documents into chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\\n\\n", "\\n", " ", ""]
#     )
#     chunks = text_splitter.split_documents(all_documents)
#     print(f"   Created {len(chunks)} chunks")
    
#     # Generate embeddings
#     print("🧠 Generating embeddings...")
#     texts = [chunk.page_content for chunk in chunks]
#     embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
#     # Prepare data for ChromaDB
#     print("💿 Storing in vector database...")
#     ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
#     documents_text = [chunk.page_content for chunk in chunks]
#     metadatas = [chunk.metadata for chunk in chunks]
    
#     # Store in ChromaDB
#     collection.add(
#         ids=ids,
#         embeddings=embeddings.tolist(),
#         documents=documents_text,
#         metadatas=metadatas
#     )
    
#     print(f"✅ Successfully processed and stored {len(chunks)} chunks")
#     print(f"📊 Total documents in database: {collection.count()}")
    
#     return True

# if __name__ == "__main__":
#     process_documents()
# """

# # Save processing script
# if not os.path.exists("process_documents.py"):
#     with open("process_documents.py", "w", encoding="utf-8") as f:
#         f.write(processing_script)
    
#     st.sidebar.info("📁 Created `process_documents.py` for document processing")

# # Run the app
# if __name__ == "__main__":
#     main()

# app.py - With Document Upload Feature
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

# Page configuration
st.set_page_config(
    page_title="Document Q&A with Upload",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .assistant-msg {
        background: white;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 70%;
        margin-right: auto;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .file-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 40px 20px;
        text-align: center;
        background: #f8fff8;
        margin: 20px 0;
        transition: all 0.3s;
    }
    .upload-area:hover {
        background: #f0fff0;
        border-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

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

# Main App
def main():
    st.title("📚 Document Q&A with Upload")
    st.markdown("Upload PDFs and ask questions about them in real-time!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Initialize system
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                if initialize_system():
                    st.success("✅ System initialized!")
                    st.rerun()
        else:
            st.success("✅ System Ready")
            
            # Show database stats
            try:
                doc_count = st.session_state.vectorstore.count()
                st.metric("Documents in DB", doc_count)
            except:
                st.metric("Documents in DB", 0)
            
            st.markdown("---")
            
            # Clear database option
            st.subheader("Database Management")
            if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                if clear_vector_database():
                    st.success("Database cleared!")
                    st.rerun()
            
            st.markdown("---")
            
            # Settings
            st.subheader("Settings")
            top_k = st.slider("Reference chunks", 1, 10, 3)
            show_sources = st.checkbox("Show sources", value=True)
            
            # Clear chat
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        st.caption("Powered by ChromaDB + Groq LLM")
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Documents")
        
        # Upload section
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag & drop or click to upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_files:
            # Show uploaded files
            st.markdown("### Uploaded Files:")
            for uploaded_file in uploaded_files:
                st.markdown(f"""
                <div class="file-card">
                    <strong>📄 {uploaded_file.name}</strong><br>
                    <small>Size: {uploaded_file.size / 1024:.1f} KB</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Process button
            if st.button("⚡ Process & Add to Database", type="primary", use_container_width=True):
                if not st.session_state.system_initialized:
                    st.error("Please initialize system first")
                elif uploaded_files:
                    with st.spinner("Processing documents..."):
                        result = process_uploaded_pdfs(uploaded_files)
                        
                        if result["success"]:
                            st.markdown(f'<div class="success-box">{result["message"]}</div>', unsafe_allow_html=True)
                            
                            # Show file details
                            for file_info in result.get("files", []):
                                st.write(f"• {file_info['name']}: {file_info['pages']} pages → {file_info['chunks']} chunks")
                            
                            st.success(f"✅ Total chunks added: {result['total_chunks']}")
                            st.rerun()
                        else:
                            st.error(result["message"])
        
        # Information box
        st.markdown("""
        <div class="info-box">
        <strong>💡 How it works:</strong>
        <ol>
        <li>Upload PDF files</li>
        <li>Click "Process & Add to Database"</li>
        <li>Ask questions about the documents</li>
        <li>Get AI-powered answers with sources</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💬 Ask Questions")
        
        if not st.session_state.system_initialized:
            st.info("👈 Please initialize the system from sidebar first")
        else:
            # Chat container
            chat_container = st.container()
            
            with chat_container:
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="user-msg"><strong>You:</strong> {message["content"]}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-msg"><strong>Assistant:</strong> {message["content"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Show sources
                        if show_sources and message.get("sources"):
                            with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                                for i, source in enumerate(message["sources"]):
                                    source_name = source.get('metadata', {}).get('source_file', 'Unknown')
                                    score = source.get('similarity_score', 0)
                                    
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.markdown(f"**{source_name}**")
                                    with col_b:
                                        st.markdown(f"*Relevance: {score:.1%}*")
                                    
                                    st.markdown(f"*{source['content']}*")
                                    st.markdown("---")
            
            # Chat input
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
            
            # Quick questions
            st.markdown("### 💡 Quick Questions")
            quick_col1, quick_col2 = st.columns(2)
            
            with quick_col1:
                if st.button("📋 Summarize", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": "Provide a summary of all uploaded documents"
                    })
                    st.rerun()
                
                if st.button("🔑 Key Points", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": "What are the key points from the documents?"
                    })
                    st.rerun()
            
            with quick_col2:
                if st.button("📖 Main Topics", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": "What are the main topics covered?"
                    })
                    st.rerun()
                
                if st.button("❓ Ask Anything", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": "What can you tell me about these documents?"
                    })
                    st.rerun()
    
    # Footer
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_b:
        st.markdown("<div style='text-align: center; color: #666;'>Powered by RAG Technology</div>", 
                   unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()