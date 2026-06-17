import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client=OpenAI()



#vector embedding
embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-large"
)

"""vector_db= QdrantVectorStore.from_existing_collection(
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

#take user input

user_query=input("Ask About Apoorv Singh : ")

search_result = vector_db.similarity_search(query=user_query) 

context = "\n\n\n".join(
    [
        f"Page content: {result.page_content}\nPage: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in search_result
    ]
)

system_prompt= f""" You are a Helpful Ai Assistant who answers query related to Apoorv Singh based on the
avaliable context retrived from the PDF file

You should only answers the question related to Apoorv Singh

context:
{context}

"""

response= openai_client.chat.completions.create(
    model="gpt-5",
    messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_query}]
)

print(f"ans:{response.choices[0].message.content}")
