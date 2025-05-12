from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import time
import logging
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medical Research Search and RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for query request
class VectorSearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=100)
    min_score: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    include_embeddings: bool = False

# Model for RAG query
class RAGQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=20)
    filters: Optional[Dict[str, Any]] = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1500, gt=0, le=4000)
    include_documents: bool = Field(default=True)
    model: str = "claude-3-opus-20240229"

# Model for search result
class SearchResult(BaseModel):
    id: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    doi_link: Optional[str] = None
    link: Optional[str] = None
    cleaned_text: Optional[str] = None
    score: float
    embedding: Optional[List[float]] = None

# Model for search response
class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    execution_time_ms: float

# Model for RAG response
class RAGResponse(BaseModel):
    query: str
    answer: str
    documents: Optional[List[SearchResult]] = None
    execution_time_ms: float

# MongoDB connection
def get_db():
    """Create MongoDB connection"""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise HTTPException(status_code=500, detail="MongoDB connection string not set")
    
    client = MongoClient(mongo_uri)
    db = client[os.getenv("MONGO_DB", "medical_research")]
    return db
    
    
    
@app.post("/rag", response_model=RAGResponse)
async def rag_answer(query: RAGQuery, db = Depends(get_db)):
        """
        Perform RAG (Retrieval-Augmented Generation) using Anthropic Claude
        """
        start_time = time.time()
        
        try:
            # Step 1: Perform search to retrieve relevant documents
            collection = db[os.getenv("MONGO_COLLECTION", "pubmed_articles")]
            
            search_pipeline = [
                {
                    "$search": {
                        "index": os.getenv("VECTOR_SEARCH_INDEX", "pubmed_vector_index"),
                        "text": {
                            "query": query.query,
                            "path": {
                                "wildcard": "*"
                            },
                        }
                    }
                },
                {
                    "$project": {
                        "score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$limit": query.top_k
                }
            ]
            
            # Add filter stage if filters are provided
            if query.filters:
                search_pipeline.append({
                    "$match": query.filters
                })
            
            # Execute the search pipeline
            results = list(collection.aggregate(search_pipeline))
            
            # Format the search results and prepare context for Claude
            formatted_results = []
            context_chunks = []
            
            for idx, result in enumerate(results):
                # Get the document ID
                doc_id = result["_id"]
                
                # Fetch the full document from the collection
                full_doc = collection.find_one({"_id": doc_id})
                
                if full_doc:
                    # Create a result with the correct field mappings
                    search_result = {
                        "id": str(doc_id),
                        "title": full_doc.get("Title", ""),
                        "authors": full_doc.get("Authors", ""),
                        "abstract": full_doc.get("Abstract", ""),
                        "publication_date": str(full_doc.get("Publication Date", "")),
                        "journal": full_doc.get("Journal", ""),
                        "doi": full_doc.get("DOI", ""),
                        "doi_link": full_doc.get("DOI Link", ""),
                        "link": full_doc.get("Link", ""),
                        "cleaned_text": full_doc.get("Cleaned Text", ""),
                        "score": result.get("score", 0.0),
                    }
                    
                    formatted_results.append(search_result)
                    
                    # Create a context chunk for this document
                    context_chunk = f"DOCUMENT {idx+1} [Score: {result.get('score', 0):.4f}]\n"
                    context_chunk += f"Title: {full_doc.get('Title', 'No title')}\n"
                    
                    if full_doc.get('Authors'):
                        context_chunk += f"Authors: {full_doc.get('Authors')}\n"
                    
                    if full_doc.get('Journal'):
                        context_chunk += f"Journal: {full_doc.get('Journal')}\n"
                    
                    if full_doc.get('Publication Date'):
                        context_chunk += f"Date: {full_doc.get('Publication Date')}\n"
                    
                    if full_doc.get('DOI'):
                        context_chunk += f"DOI: {full_doc.get('DOI')}\n"
                    
                    # Add abstract - this is crucial for medical papers
                    context_chunk += f"Abstract: {full_doc.get('Abstract', 'No abstract')}\n"
                    
                    # Add cleaned text if available
                    if "Cleaned Text" in full_doc and full_doc["Cleaned Text"]:
                        context_chunk += f"Content: {full_doc['Cleaned Text']}\n"
                    # If cleaned text is not available but full text is, use a portion of full text
                    elif "Full Text" in full_doc and full_doc["Full Text"]:
                        # Limit to first 3000 characters to avoid token limits
                        full_text = full_doc["Full Text"]
                        if len(full_text) > 3000:
                            # Try to break at paragraph or sentence
                            break_point = full_text[:3000].rfind('\n\n')
                            if break_point == -1:
                                break_point = full_text[:3000].rfind('. ')
                                if break_point != -1:
                                    break_point += 1  # Include the period
                            if break_point == -1:
                                break_point = 3000
                            context_chunk += f"Content: {full_text[:break_point]}...\n"
                        else:
                            context_chunk += f"Content: {full_text}\n"
                    
                    context_chunks.append(context_chunk)
            
            # Step 2: Check if we have any results
            if not context_chunks:
                return {
                    "query": query.query,
                    "answer": "No relevant medical documents were found for your query.",
                    "documents": [] if query.include_documents else None,
                    "execution_time_ms": (time.time() - start_time) * 1000
                }
            
            # Step 3: Combine context chunks with clear separation
            context = "\n" + "="*50 + "\n".join(context_chunks) + "\n" + "="*50 + "\n"
            
            # Step 4: Generate answer using Anthropic Claude
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                return {
                    "query": query.query,
                    "answer": "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable to enable RAG answers.",
                    "documents": formatted_results if query.include_documents else None,
                    "execution_time_ms": (time.time() - start_time) * 1000
                }
            
            try:
                # Import Anthropic client
                from anthropic import AsyncAnthropic
                
                # Create Anthropic client
                client = AsyncAnthropic(api_key=anthropic_api_key)
                
                system_prompt = f"""You are HospiAgent's Medical Search Agent for Indian healthcare professionals. Answer directly without mentioning your role or how you formed your answer.
    
    IMPORTANT GUIDELINES:
    1. Use the provided medical literature as your primary source, but you may draw on prior knowledge to enhance and complement the evidence.
    2. Focus on practical, actionable information for Indian healthcare professionals.
    3. Consider the Indian context: local disease prevalence, available medications, treatment guidelines, healthcare resources, and cultural factors.
    
    4. For any drug interaction queries, structure your response EXACTLY as follows:
       * **[MUST HAVE]** Multi-drug interaction analysis
       * **[MUST HAVE]** Severity classification of interactions
       * **[MUST HAVE]** Mechanism of interaction explanation
       * **[MUST HAVE]** Alternative medication suggestions
       * **[MUST HAVE]** Patient-specific factors consideration (age, renal function, etc.)
    
    5. For non-drug interaction queries, structure your response with:
       - CLINICAL ANSWER: Direct, concise answer to the question
       - EVIDENCE SUMMARY: Brief synthesis of key findings from the literature
       - INDIAN CONTEXT: Specific considerations for Indian healthcare settings
       - KEY TAKEAWAYS: 3-5 bullet points with the most clinically relevant information
    
    DO NOT cite document numbers, mention insufficient context, or include meta-commentary about your answer. Start directly with the content in the required format."""
    
                # Detect if query is about drug interactions
                drug_interaction_keywords = [
                    "drug interaction", "drug-drug", "interaction", "contraindication", 
                    "concomitant use", "co-administration", "drug combination", 
                    "併用", "polypharmacy", "medication interaction", "interact with"
                ]
                
                is_drug_interaction_query = any(keyword in query.query.lower() for keyword in drug_interaction_keywords)
                
                # User prompt with context and hints about format if it's a drug interaction query
                user_prompt = f"""QUESTION: {query.query}
    
    Please use the following documents as your primary source of information:
    
    {context}
    
    {"This appears to be a question about drug interactions. Make sure to follow the required format with all the [MUST HAVE] sections specified in your guidelines." if is_drug_interaction_query else ""}
    
    DO NOT include any introductory statements about synthesizing information or using documents. DO NOT cite document numbers. Start directly with your answer in the required format. If the documents don't contain all the necessary information, use your medical knowledge to provide a complete answer without mentioning gaps in the provided context."""
                
                # Call Claude API
                messages_response = await client.messages.create(
                    model=query.model,
                    max_tokens=query.max_tokens,
                    temperature=query.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                # Extract answer
                answer = messages_response.content[0].text
                
                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Return response
                return {
                    "query": query.query,
                    "answer": answer,
                    "documents": formatted_results if query.include_documents else None,
                    "execution_time_ms": execution_time_ms
                }
                
            except Exception as e:
                logger.error(f"Error generating answer with Claude: {str(e)}")
                return {
                    "query": query.query,
                    "answer": f"Error generating answer with Claude: {str(e)}",
                    "documents": formatted_results if query.include_documents else None,
                    "execution_time_ms": (time.time() - start_time) * 1000
                }
        
        except Exception as e:
            logger.error(f"Error in RAG: {str(e)}")
            raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")
            
            
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "api_version": "1.0.0"}

@app.get("/inspect-collection")
async def inspect_collection(db = Depends(get_db)):
    """
    Inspect the collection to understand its structure
    """
    try:
        collection = db[os.getenv("MONGO_COLLECTION", "pubmed_articles")]
        
        # Get total count
        total_count = collection.count_documents({})
        
        # Get a sample document
        sample_doc = collection.find_one()
        
        # Get field names and their types
        field_info = {}
        if sample_doc:
            for field, value in sample_doc.items():
                field_info[field] = type(value).__name__
        
        # Check if embedding field exists
        has_embedding = "embedding" in field_info
        embedding_dimensions = None
        if has_embedding and isinstance(sample_doc["embedding"], list):
            embedding_dimensions = len(sample_doc["embedding"])
        
        # Return collection info
        return {
            "collection_name": os.getenv("MONGO_COLLECTION", "pubmed_articles"),
            "document_count": total_count,
            "fields": field_info,
            "has_embedding": has_embedding,
            "embedding_dimensions": embedding_dimensions,
            "sample_document": {k: str(v)[:100] + "..." if isinstance(v, (str, list)) and len(str(v)) > 100 else v 
                              for k, v in sample_doc.items()} if sample_doc else None
        }
    except Exception as e:
        logger.error(f"Error inspecting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Collection inspection failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)