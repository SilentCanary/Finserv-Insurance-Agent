# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:18:20 2025

@author: advit
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import tempfile
import requests
from query_final import query_pipeline
from utils import download_pdf_and_chunk
from dotenv import load_dotenv
import os
router = APIRouter()
bearer = HTTPBearer()


load_dotenv()

VALID_TOKEN = os.getenv("VALID_TOKEN")

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return credentials.credentials

@router.post("/hackrx/run", response_model=QAResponse)
def run_qa(payload: QARequest, token: str = Depends(validate_token)):
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    response = requests.get(payload.documents)
    temp_file.write(response.content)
    temp_file.close()

    download_pdf_and_chunk(temp_file.name)

    answers = []
    for question in payload.questions:
        answer = query_pipeline(question)  
        answers.append(answer)

    return {"answers": answers}

