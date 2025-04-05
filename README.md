# Conversational RAG
#### Description:

## About The Project

**Conversational RAG Chatbot** is a Python-based chatbot application that leverages Retrieval-Augmented Generation (RAG) to answer questions about academic papers. This particular version uses the "Attention Is All You Need" paper from arXiv as the knowledge source. It employs LangGraph to orchestrate the RAG flow, and utilizes Grok API with the LLaMA3-8B model for generating responses. The app combines HuggingFace embeddings with a Chroma vector store to retrieve relevant document chunks when needed.

The system supports memory and chat history using LangGraph's checkpointer, allowing for contextual multi-turn conversations.

## Folder Contents
- **app.py**: Main file containing the LangGraph graph setup, tool routing logic, vector store creation, and chatbot logic.
- **.env**: Stores sensitive keys such as the Grok API key and other credentials.
- **/chrom_db**: Stores vector embedding of chucked document.
- **requirements.txt**: All ```pip```-installable libraries used for this project are listed here.
- **.gitignore**: Specifies intentionally untracked files to ignore.

## How It Works
- The paper "Attention Is All You Need" is loaded using the arxiv loader.
- The paper is chunked and stored in Chroma DB using HuggingFace embeddings.
- A LangGraph-based flow is used:
  - generate: The start node that handles user queries.
  - router (optional): Decides whether to invoke the retrieval tool.
  - tool node: Retrieves the most relevant documents from Chroma DB.
  - generate: Generates a final response using LLaMA3-8B with or without RAG.
  - checkpointer: Persists chat history across turns.
  - generate: Generates a final response using LLaMA3-8B with or without RAG.
  - checkpointer: Persists chat history across turns.
