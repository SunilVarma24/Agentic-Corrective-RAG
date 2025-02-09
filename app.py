import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from operator import itemgetter
from typing import List
from typing_extensions import TypedDict
import tempfile
import os

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Graph State for Agentic Corrective RAG
class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]

# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def initialize_components():
    """Initialize all components and store in session state"""
    if not st.session_state.initialized:
        st.session_state.embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        st.session_state.tv_search = TavilySearchResults(max_results=3, search_depth='advanced')

        # Initialize prompts and chains
        st.session_state.doc_grader = (
            ChatPromptTemplate.from_messages([
                ("system", "You are an expert grader assessing the relevance of a retrieved document to a user question."),
                ("human", """Retrieved document:\n{document}\nUser question:\n{question}\n
                         Respond 'yes' if the document is directly relevant to the question, otherwise respond 'no'."""),
            ])
            | st.session_state.llm.with_structured_output(GradeDocuments)
        )

        st.session_state.question_rewriter = (
            ChatPromptTemplate.from_messages([
                ("system", """Rewrite the following query to make it more specific, actionable, and optimized for search engines.
                Focus on clarity and conciseness while addressing the user's intent. Generate only one refined query."""),
                ("human", "Original question:\n{question}\nRewritten query:"),
            ])
            | st.session_state.llm
            | StrOutputParser()
        )

        st.session_state.qa_rag_chain = (
            {
                "context": itemgetter('context') | RunnableLambda(format_docs),
                "question": itemgetter('question'),
            }
            | ChatPromptTemplate.from_template("""
                Use the following context to answer the question if possible:
                {context}\nQuestion: {question}\n
                If the context doesn't answer the question, acknowledge it and recommend external resources or perform a web search.
            """)
            | st.session_state.llm
            | StrOutputParser()
        )

        st.session_state.initialized = True

def configure_retriever(uploaded_files):
    """Configure the document retriever"""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, add_start_index=True)
    chunked_docs = splitter.split_documents(docs)

    # Create Chroma Vector DB
    chroma_db = Chroma.from_documents(
        documents=chunked_docs,
        collection_name='rag_db',
        embedding=st.session_state.embed_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./rag_db"
    )

    return chroma_db.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs={"k": 3, "lambda_mult": 0.5, "score_threshold": 0.7}
    )

def format_docs(docs):
    """Format documents for the LLM"""
    return "\n\n".join(doc.page_content for doc in docs)

# Define CRAG nodes
def retrieve(state):
    """Retrieve relevant documents"""
    question = state["question"]
    documents = st.session_state.retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    """Grade document relevance"""
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"

    if documents:
        for d in documents:
            score = st.session_state.doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            st.write(f"Document: {d.page_content[:200]}... Grade: {score.binary_score}")
            if score.binary_score.strip().lower() == "yes":
                filtered_docs.append(d)
            else:
                web_search_needed = "Yes"
    else:
        web_search_needed = "Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_needed": web_search_needed
    }

def rewrite_query(state):
    """Rewrite the query for better search results"""
    question = state["question"]
    better_question = st.session_state.question_rewriter.invoke({"question": question})
    st.write(f"Rewritten Query: {better_question}")
    return {"question": better_question, "documents": state["documents"]}

def web_search(state):
    """Perform web search"""
    question = state["question"]
    documents = state["documents"]

    try:
        # Perform web search and handle results
        results = st.session_state.tv_search.invoke(question)
        search_content = []

        for result in results:
            st.write(f"Search Result: {result}")
            if isinstance(result, dict) and 'content' in result:
                search_content.append(result['content'])
            elif isinstance(result, str):
                search_content.append(result)

        if search_content:
            search_docs = Document(page_content="\n\n".join(search_content))
            documents.append(search_docs)
    except Exception as e:
        st.error(f"Web search error: {str(e)}")

    return {"documents": documents, "question": question}

def generate_answer(state):
    """Generate the final answer"""
    question = state["question"]
    documents = state["documents"]
    generation = st.session_state.qa_rag_chain.invoke(
        {"context": documents, "question": question}
    )
    return {"documents": documents, "question": question, "generation": generation}

def decide_to_generate(state):
    """Decide whether to generate answer or rewrite query"""
    return "rewrite_query" if state["web_search_needed"] == "Yes" else "generate_answer"

def build_crag():
    """Build the CRAG agent graph"""
    agentic_rag = StateGraph(GraphState)

    # Add nodes
    agentic_rag.add_node("retrieve", retrieve)
    agentic_rag.add_node("grade_documents", grade_documents)
    agentic_rag.add_node("rewrite_query", rewrite_query)
    agentic_rag.add_node("web_search", web_search)
    agentic_rag.add_node("generate_answer", generate_answer)

    # Build graph
    agentic_rag.set_entry_point("retrieve")
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
    )
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("generate_answer", END)

    return agentic_rag.compile()

def main():
    st.title("Corrective RAG (CRAG)")
    st.write("Upload your documents, ask a question, and get precise answers with corrective retrieval and generation.")

    # Initialize components
    initialize_components()

    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Query input
    query = st.text_input("Enter your query:")

    if uploaded_files and query:
        try:
            with st.spinner("Processing your request..."):
                # Configure retriever
                st.session_state.retriever = configure_retriever(uploaded_files)

                # Build and invoke CRAG
                agentic_rag = build_crag()
                response = agentic_rag.invoke({"question": query})

                # Display results
                st.subheader("Answer:")
                st.write(response['generation'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
