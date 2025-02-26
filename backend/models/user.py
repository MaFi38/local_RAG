from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class UserRegister(BaseModel):
    email: str
    password: str


