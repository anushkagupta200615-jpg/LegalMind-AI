import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma

def main():
    bare_acts_dir = './data/bare_acts/'
    
    try:
        # 1. Initialize persistent ChromaDB client
        client = chromadb.PersistentClient(path='./vectordb/chroma_store')
        
        # 8. Handle FileNotFoundError gracefully
        if not os.path.exists(bare_acts_dir) or not os.listdir(bare_acts_dir):
            print(f"No PDFs found in '{bare_acts_dir}'. Please add some PDFs first.")
            return

        documents = []
        # 2. Load all PDF files from the folder
        for file in os.listdir(bare_acts_dir):
            if file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(bare_acts_dir, file))
                documents.extend(loader.load())
                
        if not documents:
            print("No documents could be loaded.")
            return

        # 3. Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # 4. Embed chunks
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 5. Store embeddings in ChromaDB collection named 'legal_docs'
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name='legal_docs',
            persist_directory='./vectordb/chroma_store',
            client=client
        )
        
        # 6. Print a success message with the total number of chunks stored
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB collection 'legal_docs'.")
        
    except FileNotFoundError:
        print(f"Error: The directory '{bare_acts_dir}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 7. Add a __main__ guard so it runs as a standalone script
if __name__ == "__main__":
    main()
