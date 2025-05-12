import os
import time
import asyncio
import json
from functools import lru_cache
from typing import Dict, List, Optional, Any
from fastapi import Depends, HTTPException
# Remove Anthropic import
# from anthropic import AsyncAnthropic
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
import boto3
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi.applications import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Global Bedrock client
_bedrock_client = None

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
    model: str = "anthropic.claude-3-opus-20240229-v1:0"  # Updated for Bedrock model ID format

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
    
# Setup logging
logger = logging.getLogger(__name__)

# Configure environment variables with defaults to reduce lookups
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "pubmed_articles")
VECTOR_SEARCH_INDEX = os.getenv("VECTOR_SEARCH_INDEX", "pubmed_vector_index")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# Get or create Bedrock client
def get_bedrock_client():
    """Get or create a cached AWS Bedrock client"""
    global _bedrock_client
    
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            service_name="bedrock-2023-05-31",
            region_name=AWS_REGION
        )
    
    return _bedrock_client

# Query classification constants - moved outside function for performance
DRUG_INTERACTION_KEYWORDS = {"drug interaction", "drug-drug", "interaction", "contraindication", 
    "concomitant", "co-administration", "drug combination", "polypharmacy", 
    "medication interaction"}

TREATMENT_KEYWORDS = {"treatment", "therapy", "manage", "care", "intervention", "protocol", 
    "regimen", "guideline", "approach", "strategy"}

DISEASE_KEYWORDS = {"disease", "condition", "disorder", "syndrome", "pathology", "illness", 
    "symptoms", "diagnosis", "etiology"}

PUBLIC_HEALTH_KEYWORDS = {"public health", "policy", "population", "community", "prevention", 
    "screening", "surveillance", "outbreak", "epidemic", "pandemic"}

# Pre-defined system prompt - no need to recreate it for each request
SYSTEM_PROMPT = """You are HospiAgent's Medical Search Agent for Indian healthcare professionals. Answer directly without mentioning your role or how you formed your answer.

IMPORTANT GUIDELINES:
1. Use the provided medical literature as your primary source, but you may draw on prior knowledge to enhance and complement the evidence.
2. Focus on practical, actionable information for Indian healthcare professionals.
3. Consider the Indian context: local disease prevalence, available medications, treatment guidelines, healthcare resources, and cultural factors.

4. For drug interaction queries, structure your response EXACTLY as follows:
   **DRUG INTERACTION SUMMARY:**
   * **Severity:** [Critical/Major/Moderate/Minor] - Be very explicit about interaction severity
   * **Mechanism:** Clear, concise explanation of how the drugs interact
   * **Clinical Effects:** Bullet-point list of specific adverse effects or consequences
   * **Management Options:**
     - Specific dosage adjustments
     - Alternative medications
     - Monitoring parameters with specific thresholds
   * **Special Populations:** Considerations for pediatric, geriatric, pregnancy, renal/hepatic impairment
   * **Indian Context:** Availability of alternatives, local guidelines, cost considerations

5. For treatment or diagnostic queries, structure your response as follows:
   **CLINICAL ANSWER:**
   * **Key Recommendation:** 1-2 sentence direct answer
   * **Evidence Summary:** 
     - Bullet-point list of main findings from literature
     - Include efficacy rates, confidence intervals, or p-values when available
   * **Treatment Algorithm:** Step-by-step approach in numbered list format
   * **Monitoring:** Specific parameters to track with frequency and thresholds
   * **Indian Context:** 
     - Availability of treatments/diagnostics
     - Cost considerations
     - Local guidelines
   * **Red Flags:** Warning signs requiring urgent attention

6. For disease/condition queries, structure your response as follows:
   **CONDITION OVERVIEW:**
   * **Definition:** Concise clinical definition
   * **Epidemiology:** Key statistics, especially for Indian population
   * **Clinical Presentation:** Bullet-point list of symptoms and signs by frequency
   * **Diagnostic Approach:** Clear stepwise process
   * **Management:** First-line, second-line options in structured format
   * **Prevention:** Evidence-based preventive measures
   * **Special Considerations for India:** Regional variations, resource constraints

7. For public health or policy queries:
   **PUBLIC HEALTH PERSPECTIVE:**
   * **Current Status:** Data-driven summary of the situation in India
   * **Key Challenges:** Bullet-point list of barriers
   * **Evidence-Based Interventions:** Prioritized list with efficacy data
   * **Resource Optimization:** How to implement with limited resources
   * **Metrics & Evaluation:** How to measure success
   * **Policy Recommendations:** Clear, actionable steps

DO NOT cite document numbers, mention insufficient context, or include meta-commentary about your answer. Start directly with the content in the required format.

ALWAYS use bullet points, numbered lists, bold headers, and clear section breaks to make information scannable and actionable."""

# Cache for document lookups - prevents repeated DB fetches for the same documents
@lru_cache(maxsize=1000)
def get_cached_doc(doc_id_str: str, db):
    collection = db[MONGO_COLLECTION]
    from bson.objectid import ObjectId
    doc_id = ObjectId(doc_id_str)
    return collection.find_one({"_id": doc_id})

# Context processor class to handle document formatting efficiently
class ContextProcessor:
    def __init__(self, max_context_length: int = 15000):
        self.max_context_length = max_context_length
        
    def format_document(self, doc: Dict[str, Any], idx: int, score: float) -> tuple:
        """Format a document for both context and result output efficiently"""
        search_result = {
            "id": str(doc["_id"]),
            "title": doc.get("Title", ""),
            "authors": doc.get("Authors", ""),
            "abstract": doc.get("Abstract", ""),
            "publication_date": str(doc.get("Publication Date", "")),
            "journal": doc.get("Journal", ""),
            "doi": doc.get("DOI", ""),
            "doi_link": doc.get("DOI Link", ""),
            "link": doc.get("Link", ""),
            "score": score,
        }
        
        # Create context chunk efficiently with string builder
        context_parts = [
            f"DOCUMENT {idx+1} [Score: {score:.4f}]",
            f"Title: {doc.get('Title', 'No title')}"
        ]
        
        if doc.get('Authors'):
            context_parts.append(f"Authors: {doc.get('Authors')}")
        
        if doc.get('Journal'):
            context_parts.append(f"Journal: {doc.get('Journal')}")
        
        if doc.get('Publication Date'):
            context_parts.append(f"Date: {doc.get('Publication Date')}")
        
        if doc.get('DOI'):
            context_parts.append(f"DOI: {doc.get('DOI')}")
        
        # Always include abstract for medical papers
        context_parts.append(f"Abstract: {doc.get('Abstract', 'No abstract')}")
        
        # Add content using the most efficient available field
        content = None
        if "Cleaned Text" in doc and doc["Cleaned Text"]:
            content = doc["Cleaned Text"]
        elif "Full Text" in doc and doc["Full Text"]:
            content = doc["Full Text"]
        
        if content:
            # More efficient text truncation
            if len(content) > 3000:
                # Try to break at paragraph
                break_point = content.rfind('\n\n', 0, 3000)
                if break_point == -1:
                    # Try to break at sentence
                    break_point = content.rfind('. ', 0, 3000)
                    if break_point != -1:
                        break_point += 1  # Include the period
                if break_point == -1:
                    break_point = 3000
                
                context_parts.append(f"Content: {content[:break_point]}...")
            else:
                context_parts.append(f"Content: {content}")
        
        return search_result, "\n".join(context_parts)
    
    def process_search_results(self, results: List[Dict], db) -> tuple:
        """Process search results and return formatted results and context chunks"""
        formatted_results = []
        context_chunks = []
        total_context_length = 0
        
        for idx, result in enumerate(results):
            # Get document ID
            doc_id = result["_id"]
            doc_id_str = str(doc_id)
            score = result.get("score", 0.0)
            
            # Fetch document from cache or database
            try:
                full_doc = get_cached_doc(doc_id_str, db)
            except Exception:
                # If caching fails, fall back to direct fetch
                collection = db[MONGO_COLLECTION]
                full_doc = collection.find_one({"_id": doc_id})
            
            if full_doc:
                # Format document for both result and context
                search_result, context_chunk = self.format_document(full_doc, idx, score)
                
                formatted_results.append(search_result)
                
                # Check if adding this chunk would exceed the maximum context length
                chunk_length = len(context_chunk)
                if total_context_length + chunk_length <= self.max_context_length:
                    context_chunks.append(context_chunk)
                    total_context_length += chunk_length
                else:
                    # If we're about to exceed the limit, only include the most relevant documents
                    break
        
        return formatted_results, context_chunks

# Async helper function to execute search pipeline and process results
async def execute_search(query_text: str, filters: Dict, top_k: int, db):
    # Build optimized search pipeline
    search_pipeline = [
        {
            "$search": {
                "index": VECTOR_SEARCH_INDEX,
                "text": {
                    "query": query_text,
                    "path": {"wildcard": "*"}
                }
            }
        },
        {
            "$project": {
                "_id": 1,  # Only project the ID and score to minimize data transfer
                "score": {"$meta": "searchScore"}
            }
        },
        {"$limit": top_k}
    ]
    

    
    # Execute the search pipeline asynchronously
    collection = db[MONGO_COLLECTION]
    results = list(collection.aggregate(search_pipeline))
    
    # Process results
    processor = ContextProcessor()
    formatted_results, context_chunks = processor.process_search_results(results, db)
    
    return formatted_results, context_chunks

# Helper function to determine query type and format hint
def get_query_type_and_hint(query_text: str) -> str:
    """Efficiently determine query type and return appropriate format hint"""
    query_lower = query_text.lower()
    
    # Check for query types using set intersection for efficiency
    query_words = set(query_lower.split())
    
    if query_words.intersection(DRUG_INTERACTION_KEYWORDS):
        return "This appears to be a drug interaction query. Follow the DRUG INTERACTION SUMMARY format with all sections."
    elif query_words.intersection(TREATMENT_KEYWORDS):
        return "This appears to be a treatment query. Follow the CLINICAL ANSWER format with all sections."
    elif query_words.intersection(DISEASE_KEYWORDS):
        return "This appears to be a disease/condition query. Follow the CONDITION OVERVIEW format with all sections."
    elif query_words.intersection(PUBLIC_HEALTH_KEYWORDS):
        return "This appears to be a public health query. Follow the PUBLIC HEALTH PERSPECTIVE format with all sections."
    
    return "Structure your response with clear headers, bullet points, and numbered lists for each major section."

# Connection pool for database
@asynccontextmanager
async def lifespan(app):
    # Initialize connection pool or other startup resources
    yield
    # Clean up resources at shutdown

# Main RAG function - MODIFIED TO USE BEDROCK
@app.post("/rag", response_model=RAGResponse)
async def rag_answer(query: RAGQuery, db = Depends(get_db)):
    """
    Perform RAG (Retrieval-Augmented Generation) using Claude via AWS Bedrock
    """
    start_time = time.time()
    
    try:
        # Step 1: Launch search asynchronously
        formatted_results, context_chunks = await execute_search(
            query_text=query.query,
            filters=query.filters,
            top_k=query.top_k,
            db=db
        )
        
        # Step 2: Check if we have any results
        if not context_chunks:
            return {
                "query": query.query,
                "answer": "No relevant medical documents were found for your query.",
                "documents": [] if query.include_documents else None,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
        
        # Step 3: Efficiently combine context chunks with minimal string operations
        context = "\n" + "="*50 + "\n" + "\n".join(context_chunks) + "\n" + "="*50 + "\n"
        
        # Step 4: Generate answer using Claude via AWS Bedrock
        try:
            # Get cached Bedrock client
            bedrock_client = get_bedrock_client()
            
            # Get query type and format hint
            format_hint = get_query_type_and_hint(query.query)
            
            # Create user prompt efficiently
            user_prompt = f"""QUESTION: {query.query}

Please use the following documents as your primary source of information:

{context}

{format_hint}

DO NOT include any introductory statements about synthesizing information or using documents. DO NOT cite document numbers. Start directly with your answer in the required format. If the documents don't contain all the necessary information, use your medical knowledge to provide a complete answer without mentioning gaps in the provided context. Use formatting extensively to make your answer scannable."""
            
            # Create the request body for Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": query.max_tokens,
                "temperature": query.temperature,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            
            # Call Bedrock API
            response = bedrock_client.invoke_model(
                modelId='anthropic.claude-3-haiku-20240307-v1:0',  # Removed ':0' suffix
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response["body"].read().decode("utf-8"))
            answer = response_body["content"][0]["text"]
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Return response efficiently
            return {
                "query": query.query,
                "answer": answer,
                "documents": formatted_results if query.include_documents else None,
                "execution_time_ms": execution_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with Bedrock Claude: {str(e)}")
            return {
                "query": query.query,
                "answer": f"Error generating answer with Bedrock Claude: {str(e)}",
                "documents": formatted_results if query.include_documents else None,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
    
    except Exception as e:
        logger.error(f"Error in RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")


# Model for Clinical Trial RAG query
class ClinicalTrialQuery(BaseModel):
    query: str
    top_k: int = Field(default=8, gt=0, le=30)  # Increased default to get more trial results
    phase: Optional[str] = None  # Filter by trial phase (I, II, III, IV)
    status: Optional[str] = None  # Filter by status (recruiting, completed, etc.)
    condition: Optional[str] = None  # Specific condition being studied
    intervention_type: Optional[str] = None  # Drug, device, procedure, etc.
    max_tokens: int = Field(default=2000, gt=0, le=4000)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    include_documents: bool = Field(default=True)

# Pre-defined system prompt specifically for clinical trials
CLINICAL_TRIAL_SYSTEM_PROMPT = """You are HospiAgent's Clinical Trial Search Agent for Indian healthcare professionals. Answer directly without mentioning your role or how you formed your answer.

IMPORTANT GUIDELINES:
1. Use the provided clinical trial literature as your primary source, but you may draw on prior knowledge to enhance and complement the evidence.
2. Focus on practical, actionable information about clinical trials for Indian healthcare professionals.
3. Consider the Indian context: local regulatory environment, standard of care, available treatments, and cultural factors.

Structure your response with the following sections:

**CLINICAL TRIAL SUMMARY:**
* **Overview:** Brief summary of relevant trials addressing the query
* **Study Design:** Details on trial design, phases, and methodology
* **Inclusion/Exclusion Criteria:** Key eligibility criteria for the trials
* **Interventions:** Description of treatments or interventions being studied
* **Outcomes:** Primary and secondary endpoints with available results if completed
* **Safety Profile:** Notable adverse events and safety considerations
* **Indian Context:** 
  - Relevance to Indian patient populations
  - Whether trials are/were conducted in India
  - Implications for Indian healthcare practices
  - Regulatory status in India if applicable
* **Clinical Implications:** How findings might impact current clinical practice
* **Limitations:** Important caveats or limitations of the trial data

DO NOT cite document numbers or mention insufficient context. Format your answer with bullet points, numbered lists, bold headers, and clear section breaks to make information scannable for busy clinicians."""

# Helper function to build clinical trial specific filters
def build_clinical_trial_filters(query: ClinicalTrialQuery) -> Dict[str, Any]:
    """Build MongoDB filter dictionary for clinical trial queries"""
    # Start with base filters that identify clinical trial documents
    filters = {
        "$or": [
            {"Title": {"$regex": "clinical trial|trial|phase|randomized|randomised", "$options": "i"}},
            {"Abstract": {"$regex": "clinical trial|phase [I|II|III|IV]|randomized|randomised|NCT[0-9]", "$options": "i"}},
            {"Keywords": {"$regex": "clinical trial", "$options": "i"}},
            {"Document Type": {"$regex": "clinical trial", "$options": "i"}}
        ]
    }
    
    # Add user-specified filters
    if query.phase:
        phase_regex = f"phase {query.phase}|phase-{query.phase}|phase{query.phase}"
        filters["$and"] = filters.get("$and", [])
        filters["$and"].append({
            "$or": [
                {"Title": {"$regex": phase_regex, "$options": "i"}},
                {"Abstract": {"$regex": phase_regex, "$options": "i"}}
            ]
        })
    
    if query.status:
        status_regex = f"{query.status}"
        filters["$and"] = filters.get("$and", [])
        filters["$and"].append({
            "$or": [
                {"Title": {"$regex": status_regex, "$options": "i"}},
                {"Abstract": {"$regex": status_regex, "$options": "i"}}
            ]
        })
    
    if query.condition:
        condition_regex = f"{query.condition}"
        filters["$and"] = filters.get("$and", [])
        filters["$and"].append({
            "$or": [
                {"Title": {"$regex": condition_regex, "$options": "i"}},
                {"Abstract": {"$regex": condition_regex, "$options": "i"}},
                {"Keywords": {"$regex": condition_regex, "$options": "i"}}
            ]
        })
    
    if query.intervention_type:
        intervention_regex = f"{query.intervention_type}"
        filters["$and"] = filters.get("$and", [])
        filters["$and"].append({
            "$or": [
                {"Title": {"$regex": intervention_regex, "$options": "i"}},
                {"Abstract": {"$regex": intervention_regex, "$options": "i"}}
            ]
        })
    
    # If the user provided additional filters, merge them
    if hasattr(query, 'filters') and query.filters:
        # Combine with existing filters using $and
        if "$and" in filters:
            filters["$and"].extend([{k: v} for k, v in query.filters.items()])
        else:
            filters["$and"] = [{k: v} for k, v in query.filters.items()]
    
    return filters

# Helper function to determine clinical trial format hint based on query
def get_clinical_trial_hint(query_text: str) -> str:
    """Determine the appropriate format hint for clinical trial queries"""
    query_lower = query_text.lower()
    
    if any(term in query_lower for term in ["safety", "adverse", "side effect", "toxicity"]):
        return "This appears to be a safety query. Focus on the Safety Profile section with detailed adverse event data."
    
    elif any(term in query_lower for term in ["efficacy", "effectiveness", "outcome", "result"]):
        return "This appears to be an efficacy query. Focus on the Outcomes section with detailed efficacy data."
    
    elif any(term in query_lower for term in ["design", "methodology", "protocol", "inclusion", "exclusion"]):
        return "This appears to be a study design query. Focus on the Study Design and Inclusion/Exclusion Criteria sections."
    
    elif any(term in query_lower for term in ["india", "indian", "local", "regional"]):
        return "This appears to be a query about Indian context. Focus on the Indian Context section and regional relevance."
    
    return "Provide a balanced summary across all sections, highlighting the most relevant clinical trial information."

# New endpoint for clinical trial specific RAG
@app.post("/clinical_trials_rag", response_model=RAGResponse)
async def clinical_trials_rag(query: ClinicalTrialQuery, db = Depends(get_db)):
    """
    Perform RAG specifically for clinical trial related queries using Claude via AWS Bedrock
    """
    start_time = time.time()
    
    try:
        # Build clinical trial specific filters
        filters = build_clinical_trial_filters(query)
        
        # Step 1: Launch search asynchronously with clinical trial filters
        formatted_results, context_chunks = await execute_search(
            query_text=query.query,
            filters=filters,
            top_k=query.top_k,
            db=db
        )
        
        # Step 2: Check if we have any results
        if not context_chunks:
            return {
                "query": query.query,
                "answer": "No relevant clinical trial documents were found for your query. Consider broadening your search terms or checking for alternative terminology.",
                "documents": [] if query.include_documents else None,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
        
        # Step 3: Efficiently combine context chunks with minimal string operations
        context = "\n" + "="*50 + "\n" + "\n".join(context_chunks) + "\n" + "="*50 + "\n"
        
        # Step 4: Generate answer using Claude via AWS Bedrock
        try:
            # Get cached Bedrock client
            bedrock_client = get_bedrock_client()
            
            # Get clinical trial specific format hint
            format_hint = get_clinical_trial_hint(query.query)
            
            # Create user prompt efficiently - tailored for clinical trials
            user_prompt = f"""QUESTION ABOUT CLINICAL TRIALS: {query.query}

Please use the following clinical trial documents as your primary source of information:

{context}

{format_hint}

DO NOT include any introductory statements about synthesizing information or using documents. DO NOT cite document numbers. Start directly with your answer using the CLINICAL TRIAL SUMMARY format. If the documents don't contain all the necessary information, use your medical knowledge to provide a complete answer without mentioning gaps in the provided context. Use formatting extensively to make your answer scannable."""
            
            # Create the request body for Bedrock - using clinical trial system prompt
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": query.max_tokens,
                "temperature": query.temperature,
                "system": CLINICAL_TRIAL_SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            
            # Call Bedrock API
            response = bedrock_client.invoke_model(
                modelId='anthropic.claude-3-haiku-20240307-v1:0',  # Removed ':0' suffix
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response["body"].read().decode("utf-8"))
            answer = response_body["content"][0]["text"]
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Return response efficiently
            return {
                "query": query.query,
                "answer": answer,
                "documents": formatted_results if query.include_documents else None,
                "execution_time_ms": execution_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error generating answer with Bedrock Claude for clinical trials: {str(e)}")
            return {
                "query": query.query,
                "answer": f"Error generating answer about clinical trials: {str(e)}",
                "documents": formatted_results if query.include_documents else None,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
    
    except Exception as e:
        logger.error(f"Error in clinical trials RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clinical trials RAG failed: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)