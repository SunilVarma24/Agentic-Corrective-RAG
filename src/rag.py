# src/rag.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from operator import itemgetter

# Build the QA RAG Chain

# Prompt template for QA chain
prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If no context is present or if you don't know the answer, just say that you don't know the answer.
Do not make up the answer unless it is there in the provided context.
Give a detailed answer and to the point answer with regard to the question.
Question:
{question}
Context:
{context}
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_rag_chain = (
    {
        "context": (itemgetter('context')
                    | RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
    | prompt_template
    | __import__("src.models", fromlist=["llm"]).llm  # use the llm from models.py
    | StrOutputParser()
)

# Create a Query Rephraser
SYS_PROMPT = """Act as a question re-writer and perform the following task:
- Convert the following input question to a better version that is optimized for web search.
- When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Here is the initial question:
{question}
Formulate an improved question.
"""),
    ]
)
question_rewriter = (
    re_write_prompt
    | __import__("src.models", fromlist=["llm"]).llm
    | StrOutputParser()
)

# Define Tavily Web Search (to be used later in the graph)
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)
