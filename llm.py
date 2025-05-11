import openai
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(query: str, context: str) -> str:
    prompt = f"""
You are a clinical trial expert in India.
Context:
{context}
Question:
{query}
Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    content = response.choices[0].message.content
    # Handle the potential None case
    return content if content is not None else ""