import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
pdf_path= Path(__file__).parent / "Apoorv_Singh_AI__Engineer.pdf"

load_dotenv()

loader= PyPDFLoader(file_path=pdf_path)

docs=loader.load()


# split the docs in smaller chunks
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=250
)

chunks =text_splitter.split_documents(documents=docs)

#vector embedding

embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-large"
)


"""vector_store=QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_RAG"
)"""



vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=os.getenv("https://659ff597-a9b3-4c49-a522-5d054b5dc118.us-west-1-0.aws.cloud.qdrant.io"),
    api_key=os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjEwZTQzODAtMjE2MS00MjlkLWE3NjktZTM1MDNkODVmZjIwIn0._ocfkW9VvGhuQA8To1TzA7gWtgxHxhpL1JJ52UOVh3g"),
    collection_name="learning_RAG"
)

print("indexing of document completed...")
