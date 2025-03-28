# main.py

from IPython.display import display, Markdown
from src.document import load_and_chunk_document, create_chroma_retriever
from src.models import embed_model
from src.graph import get_agentic_rag, display_graph

def main():
    # Load & chunk the document
    file_path = "/content/practitioners_guide_to_mlops_whitepaper.pdf"
    chunked_docs = load_and_chunk_document(file_path)
    
    # Create the vector DB retriever using the chunked document and embedding model
    retriever = create_chroma_retriever(chunked_docs, embed_model)
    
    # Build the agentic corrective RAG graph (it uses the retriever within the retrieve() function)
    agentic_rag = get_agentic_rag()
    
    # Optionally display the graph
    display_graph()
    
    # Test the Agentic CRAG System with a sample query
    query = "what is the difference between mlops vs llmops?"
    response = agentic_rag.invoke({"question": query})
    display(Markdown(response['generation']))

if __name__ == "__main__":
    main()
