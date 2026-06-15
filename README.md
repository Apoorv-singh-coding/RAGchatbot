Project: Resume Chatbot using RAG (Retrieval-Augmented Generation)
Overview

Developed an AI-powered Resume Chatbot that allows users to ask questions about my professional background in natural language. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from my resume and generate accurate, context-aware responses using a Large Language Model (LLM).

Problem Statement

Traditional resumes are static documents that require recruiters to manually search for relevant information. The Resume Chatbot enables recruiters and hiring managers to interact with the resume conversationally and instantly obtain specific information about skills, experience, projects, education, and achievements.

Tech Stack
Python
LangChain
OpenAI GPT Models
OpenAI Embeddings (text-embedding-3-large)
Qdrant Vector Database
FastAPI / Streamlit (if used)
Docker
PyPDF
Python Dotenv
Architecture
Document Ingestion
Resume PDF is loaded using PyPDFLoader.
Text is extracted from the document.
Text Chunking
Resume content is split into smaller chunks using RecursiveCharacterTextSplitter.
Chunk Size: 500 characters
Chunk Overlap: 250 characters
Embedding Generation
Each chunk is converted into vector embeddings using OpenAI Embeddings.
Semantic meaning of text is preserved in vector form.
Vector Storage
Embeddings are stored in Qdrant Vector Database.
Enables fast semantic search and similarity matching.
User Query Processing
User asks a question such as:
"What are Apoorv's technical skills?"
"Tell me about his AI projects."
"How many years of experience does he have?"
Retrieval
Similarity search is performed in Qdrant.
Most relevant resume chunks are retrieved.
Augmented Generation
Retrieved context is appended to the user's question.
GPT model generates an accurate answer based only on resume content.
Key Features
Conversational Resume Assistant
Semantic Search
Context-Aware Responses
Hallucination Reduction using RAG
Fast Retrieval using Qdrant
Recruiter-Friendly Interface
Dockerized Deployment
