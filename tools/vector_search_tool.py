import os
import chromadb
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings

@tool
def legal_vector_search(query: str) -> str:
    """
    Searches Indian legal documents (IPC, CRPC, IT Act, etc.) stored in the local vector DB.
    Returns the top 4 most relevant chunks with source document name and page number.
    """
    persist_directory = './vectordb/chroma_store'
    
    if not os.path.exists(persist_directory):
        return 'Vector DB not initialized. Run vectordb/ingest.py first.'
        
    try:
        # Initialize client fresh each call for statelessness
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Check if collection exists
        try:
            collection = client.get_collection('legal_docs')
        except Exception:
            return 'Vector DB not initialized. Run vectordb/ingest.py first.'
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        query_embedding = embeddings.embed_query(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant legal documents found."
            
        formatted_output = ""
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            
            source = metadata.get('source', 'Unknown Source')
            page = metadata.get('page', 'Unknown Page')
            
            # Max 300 chars for text chunk
            chunk_snippet = doc_text[:300] + ("..." if len(doc_text) > 300 else "")
            
            formatted_output += f"{i+1}. Source: {source} (Page {page})\nText: {chunk_snippet}\n\n"
            
        return formatted_output.strip()
        
    except Exception as e:
        return f"Error accessing Vector DB: {str(e)}"
