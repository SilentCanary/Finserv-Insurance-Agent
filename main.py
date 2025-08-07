# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:17:25 2025

@author: advit
"""

from fastapi import FastAPI
import uvicorn
from router import router
import nest_asyncio
nest_asyncio.apply()

app = FastAPI(title="LLM Doc QA Backend", version="1.0")
app.include_router(router, prefix="/api/v1")
