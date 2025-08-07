# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 20:46:44 2025

@author: advit
"""

import re
import faiss
import json
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import torch
from functools import lru_cache
import requests
import os
from dotenv import load_dotenv

GROQ_API_KEY = os.getenv("GROQ_KEY")


embedder = SentenceTransformer("./all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')

index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/chunks.pkl", "rb") as f:
    doc_chunks, sources = pickle.load(f)

def groq_call(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content'].strip()

@lru_cache(maxsize=128)
def reformulate_query(original_query):
    prompt = f"""
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
    ALWAYS convert fragmentary notes like “34F, hotel issue Thailand” into a full question like “Does the policy cover hotel rebooking expenses in Thailand?”
    But keep the query concise and don't add any unnecessary assumptions.

    Original query: {original_query}

    Rewritten query:
    """
    return groq_call(prompt).strip()

@lru_cache(maxsize=128)
def cached_llm_call(prompt: str) -> str:
    return groq_call(prompt).strip()

def query_pipeline(user_query, top_k=10, rerank_k=5):
    refined_query = reformulate_query(user_query)
    query_embedding = embedder.encode([refined_query], convert_to_tensor=False, normalize_embeddings=True)
    distance, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [doc_chunks[i] for i in indices[0]]
    retrieved_sources = [sources[i] for i in indices[0]]

    pairs = [(refined_query, chunk) for chunk in retrieved_chunks]
    with torch.no_grad():
        scores = cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)

    reranked = sorted(zip(retrieved_chunks, retrieved_sources, scores), key=lambda x: x[2], reverse=True)
    top_chunks = [chunk for chunk, _, _ in reranked[:rerank_k]]

    answer_prompt = f"""
You are an expert assistant helping users understand their health insurance coverage. Use the following retrieved document snippets to answer the user's question.

Answer clearly and concisely in plain English. Do not return a JSON or structured output. Just provide a helpful, factual response based on the information available.
Keep the answer concise and to the point.
User Question: "{user_query}"

Relevant Document Snippets:
{chr(10).join([f"- {chunk.strip()}" for chunk in top_chunks])}

Answer:"""


    try:
        final_answer_text = cached_llm_call(answer_prompt)
        return final_answer_text
    except Exception:
        return "Error: Failed to generate or parse response."
