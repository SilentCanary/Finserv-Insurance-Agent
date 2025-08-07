# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:20:15 2025

@author: advit
"""

from preprocessing import read_pdf, chunk_text, build_faiss_index
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./all-MiniLM-L6-v2')

def download_pdf_and_chunk(pdf_path):
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump((chunks, [pdf_path]*len(chunks)), f)
