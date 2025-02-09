# Corrective Retrieval-Augmented Generation (RAG) using Gemini 1.5 Flash

## Project Overview
This project implements a Agentic Corrective Retrieval-Augmented Generation (RAG) system using Gemini 1.5 Flash. It retrieves context from stored documents in ChromaDB and, if needed, refines the query for web search to ensure accurate responses. The system is deployed using Streamlit for an interactive user experience.

## Introduction
Traditional RAG models rely solely on pre-stored document retrieval, which can sometimes lead to incomplete or irrelevant responses. This project enhances the RAG approach by introducing **query rewriting** when retrieved documents are insufficient. The system refines the query for optimized web search and integrates the results into the response, improving overall accuracy and relevance.

## How It Works
1. **Document Processing**:
   - Users upload a document.
   - The document is chunked and stored as vector embeddings in ChromaDB.

2. **Query Processing & Retrieval**:
   - The input query is embedded and a similarity search is performed in ChromaDB.
   - If relevant results are found, they are used for response generation.

3. **Query Rewriting & Web Search**:
   - If retrieved documents are not relevant, the LLM rewrites the query for better web search optimization.
   - The new query is used to fetch relevant information from the web.

4. **Final Response Generation**:
   - The retrieved document context and web search data are combined.
   - The LLM generates a refined and contextually accurate response.
