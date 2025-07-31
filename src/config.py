import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    USER: str
    PASSWORD: str
    EXPERIMENT_ID: str
    RUN_ID: str

    class Config:
        env_file = ".env"

settings = Settings()
