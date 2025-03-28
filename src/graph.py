# src/graph.py

from langgraph.graph import END, StateGraph
from typing import List
from typing_extensions import TypedDict
from IPython.display import Image, display
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from src.rag import qa_rag_chain, question_rewriter, tv_search
from src.document import create_chroma_retriever
from src.models import llm, embed_model

# Define the Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]

# --- Retrieval & Grading Functions ---

def retrieve(state: GraphState, retriever) -> GraphState:
    """
    Retrieve documents from the vector DB based on the question.
    """
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "generation": "", "web_search_needed": "No"}

# Data model for LLM output format for grading
from langchain_core.pydantic_v1 import BaseModel, Field

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Build grader chain
structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
Follow these instructions for grading:
  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
{document}
User question:
{question}
"""),
    ]
)
doc_grader = (grade_prompt | structured_llm_grader)

def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question using an LLM grader.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"

    if documents:
        for d in documents:
            score = doc_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search_needed = "Yes"
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"

    return {"documents": filtered_docs, "question": question, "generation": "", "web_search_needed": web_search_needed}

def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrites the question to produce a better version.
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "generation": "", "web_search_needed": state["web_search_needed"]}

def web_search(state: GraphState) -> GraphState:
    """
    Performs a web search based on the rewritten question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = tv_search.invoke(question)
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question, "generation": "", "web_search_needed": state["web_search_needed"]}

def generate_answer(state: GraphState) -> GraphState:
    """
    Generates an answer from the context documents using the QA RAG chain.
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "web_search_needed": state["web_search_needed"]}

def decide_to_generate(state: GraphState) -> str:
    """
    Decides whether to generate an answer or re-write the query based on document relevance.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]
    if web_search_needed == "Yes":
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY---")
        return "rewrite_query"
    else:
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"

# --- Build the Agent Graph ---

from langgraph.graph import StateGraph

agentic_rag = StateGraph(GraphState)

# Define the nodes
agentic_rag.add_node("retrieve", retrieve)
agentic_rag.add_node("grade_documents", grade_documents)
agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("web_search", web_search)
agentic_rag.add_node("generate_answer", generate_answer)

# Build graph: set entry and add edges
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

# Compile the graph
agentic_rag = agentic_rag.compile()

# Function to display the graph (for use in main.py)
def display_graph():
    from IPython.display import Image, display
    display(Image(agentic_rag.get_graph().draw_mermaid_png()))

# Expose the compiled graph
def get_agentic_rag():
    return agentic_rag
