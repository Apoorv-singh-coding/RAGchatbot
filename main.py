from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Apoorv Singh RAG API",
    version="1.0.0"
)

# OpenAI Client
openai_client = OpenAI()

# Embedding Model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Qdrant Connection
"""vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_RAG",
    embedding=embedding_model,
)"""

vector_db = QdrantVectorStore.from_existing_collection(
    url=os.getenv("https://659ff597-a9b3-4c49-a522-5d054b5dc118.us-west-1-0.aws.cloud.qdrant.io"),
    api_key=os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjEwZTQzODAtMjE2MS00MjlkLWE3NjktZTM1MDNkODVmZjIwIn0._ocfkW9VvGhuQA8To1TzA7gWtgxHxhpL1JJ52UOVh3g"),
    collection_name="RAGChatBot",
    embedding=embedding_model,
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        # Retrieve relevant chunks
        search_result = vector_db.similarity_search(
            query=request.query,
            k=5
        )

        # Build Context
        context = "\n\n\n".join(
            [
                f"Page content: {result.page_content}\n"
                f"Page: {result.metadata.get('page_label', 'N/A')}\n"
                f"File Location: {result.metadata.get('source', 'N/A')}"
                for result in search_result
            ]
        )

        system_prompt = f"""
You are a helpful AI assistant.

Answer only questions related to Apoorv Singh using the provided context.

If the answer is not present in the context, reply:
"I could not find information about that in the provided documents."

Context:
{context}
"""

        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query},
            ],
        )

        answer = response.choices[0].message.content

        sources = [
            {
                "page": doc.metadata.get("page_label"),
                "source": doc.metadata.get("source"),
            }
            for doc in search_result
        ]

        return QueryResponse(
            answer=answer,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
