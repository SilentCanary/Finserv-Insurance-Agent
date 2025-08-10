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
from utils import download_file_and_chunk
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



def get_file_extension_fron_url(url:str) ->str:
    url=url.lower()
    if url.endswith('.pdf'):
        return '.pdf'
    elif url.endswith('.docx'):
        return '.docx'
    elif url.endswith('.eml'):
        return '.eml'
       
    else:
        try:
            resp=requests.head(url,timeout=3)
            content_type=resp.headers.get('content-type','').lower()
            if 'pdf' in content_type:
                return '.pdf'
            elif 'msword' in content_type or 'officedocument' in content_type:
                return '.docx'
            elif 'message/rfc822' in content_type or 'eml' in content_type:
                return '.eml'
        except Exception:
            pass
        return '.pdf'  #default , this will give terrible result but prevent from breaking 

@router.post("/hackrx/run", response_model=QAResponse)
def run_qa_post(payload: QARequest, token: str = Depends(validate_token)):
    
    ext=get_file_extension_fron_url(payload.documents)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    response = requests.get(payload.documents)
    temp_file.write(response.content)
    temp_file.close()

    download_file_and_chunk(temp_file.name)

    answers = []
    for question in payload.questions:
        answer = query_pipeline(question)  
        answers.append(answer)

    return {"answers": answers}



@router.get("/hackrx/run", response_model=QAResponse)
def run_qa_get(payload: QARequest, token: str = Depends(validate_token)):
    ext=get_file_extension_fron_url(payload.documents)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    response = requests.get(payload.documents)
    temp_file.write(response.content)
    temp_file.close()

    download_file_and_chunk(temp_file.name)

    answers = []
    for question in payload.questions:
        answer = query_pipeline(question)  
        answers.append(answer)

    return {"answers": answers}

