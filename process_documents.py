# process_documents_simple.py
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import uuid
import shutil

def process_documents():
    print("🚀 Starting document processing...")
    
    # Create directories if they don't exist
    os.makedirs("./data/pdfs", exist_ok=True)
    
    # Check for PDFs
    pdf_dir = Path("./data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in ./data/pdfs/")
        print("Please add your PDF files to that directory and run again.")
        return False
    
    print(f"📄 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # Initialize embedding model
    print("\n🔧 Initializing embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Clear old vector store (simplest approach)
    vector_store_path = "./data/vector_store"
    if os.path.exists(vector_store_path):
        print("🗑️ Removing old vector store...")
        try:
            shutil.rmtree(vector_store_path)
            print("   ✅ Old vector store removed")
        except Exception as e:
            print(f"   ⚠️ Could not remove old vector store: {e}")
    
    print("💾 Creating new vector database...")
    os.makedirs(vector_store_path, exist_ok=True)
    client = chromadb.PersistentClient(path=vector_store_path)
    
    # Create new collection
    collection = client.create_collection(
        name="pdf_documents",
        metadata={"description": "PDF document embeddings for RAG"}
    )
    
    all_documents = []
    
    # Load and process PDFs
    for pdf_file in pdf_files:
        print(f"\n📖 Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
            
            all_documents.extend(documents)
            print(f"   ✅ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"   ❌ Error loading {pdf_file.name}: {e}")
            continue
    
    if not all_documents:
        print("\n❌ No documents were loaded")
        return False
    
    print(f"\n📊 Total pages loaded: {len(all_documents)}")
    
    # Split documents into chunks
    print("\n✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"   Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("\n🧠 Generating embeddings...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    print("\n💿 Storing in vector database...")
    
    # Store all chunks at once
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    documents_text = [chunk.page_content for chunk in chunks]
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata['chunk_index'] = i
        metadata['content_length'] = len(chunk.page_content)
        metadatas.append(metadata)
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents_text,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully stored {len(chunks)} chunks")
    print(f"📊 Total documents in database: {collection.count()}")
    
    # Verify
    print("\n🔍 Verifying storage...")
    count = collection.count()
    if count == len(chunks):
        print(f"   ✅ Verification passed: {count} chunks stored")
    else:
        print(f"   ⚠️ Verification warning: Expected {len(chunks)}, got {count}")
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("PDF Document Processor")
    print("=" * 50)
    
    try:
        success = process_documents()
        if success:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS: Documents processed successfully!")
            print("=" * 50)
            print("\n📋 Summary:")
            print("- PDFs processed from: ./data/pdfs/")
            print("- Vector store created at: ./data/vector_store/")
            print("- Collection name: pdf_documents")
            print("\n🚀 Next steps:")
            print("1. Run: streamlit run app.py")
            print("2. Click 'Initialize System' in sidebar")
            print("3. Start asking questions!")
        else:
            print("\n" + "=" * 50)
            print("❌ FAILED: Document processing failed")
            print("=" * 50)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)