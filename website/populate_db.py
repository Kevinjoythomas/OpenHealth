import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "./chroma"
DATA_PATH = "./data"

def clear_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
def load_documents():
    document_loader  =PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap = 80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source =  chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index+=1
        else:
            current_chunk_index = 0
        last_page_id = current_page_id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        
    return chunks

def add_to_chroma(chunks: list[Document]):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embeddings
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if(new_chunks):
        print(f"✅  Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks,ids = new_chunk_ids)
        db.persist()
    else:
        print("👉 No new documents to add")
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--reset",action="store_true")
    args = parser.parse_args()
    
    if args.reset:
        print("Clearing Database")
        clear_db()
        
    documents = load_documents()
    chunks = split_docs(documents)
    add_to_chroma(chunks)
    print("\n✨ Task Completed Successfully! Your Chroma database has been updated.")
if __name__ == "__main__":
    main()