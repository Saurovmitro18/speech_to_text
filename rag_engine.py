from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Initialize local embedding model
print("Loading vector embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(transcript_blocks, persist_directory="./chroma_db"):
    """
    Takes timestamped speaker transcripts, converts them into LangChain Documents,
    and indexes them inside a local Chroma Vector Database.
    """
    if not transcript_blocks:
        return None

    documents = []
    
    # Convert transcript blocks into search-ready vector documents
    for block in transcript_blocks:
        doc = Document(
            page_content=block["text"],
            metadata={
                "start": block["start"],
                "end": block["end"],
                "speaker": block["speaker"]
            }
        )
        documents.append(doc)

    # Index into ChromaDB
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Successfully indexed {len(documents)} segments into ChromaDB.")
    return vectorstore

def search_transcript(query: str, vectorstore, top_k: int = 3):
    """
    Performs a semantic similarity search on the meeting transcript
    and returns relevant segments along with timestamps.
    """
    if not vectorstore:
        return []

    # Search vector store for top_k most relevant matches
    results = vectorstore.similarity_search(query, k=top_k)
    
    retrieved_data = []
    for doc in results:
        retrieved_data.append({
            "text": doc.page_content,
            "speaker": doc.metadata.get("speaker", "Speaker 1"),
            "timestamp": f"{doc.metadata.get('start')}s - {doc.metadata.get('end')}s"
        })
    
    return retrieved_data