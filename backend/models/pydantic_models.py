from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    LLAMA3_1 = "llama3.1"
    LLAMA3_2 = "llama3.2"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    user_id: int
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName 
    user_id : int

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    user_id: int
    file_size: int  = None
    content_type: str = None

class DeleteFileRequest(BaseModel):
    file_id: int
    user_id:int


