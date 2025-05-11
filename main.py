from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from embeddings import get_embedding
from llm import call_llm
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
collection = client["data"]["pubmed-rag"] 

# Request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_trials(payload: QueryRequest):
    try:
        query = payload.query
        query_embedding = get_embedding(query)

        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": 5,
                    "index": "embedding_index"
                }
            }
        ])
        context_docs = [doc["description"] for doc in results]
        if not context_docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        context = "\n\n".join(context_docs)
        answer = call_llm(query, context)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

