import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from src.config import settings

security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != settings.USER or credentials.password != settings.PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials
