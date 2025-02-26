from fastapi import FastAPI, File, UploadFile, HTTPException
from models.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from fastapi.security import OAuth2PasswordBearer
from models.user import UserRegister
from services.langchain_utils import get_rag_chain
from services.database import (
      insert_application_logs,
      get_chat_history, get_all_documents, 
      insert_document_record, 
      delete_document_record,
      get_user_by_email,
      insert_user,
      get_user_chat_history,
      delete_chat_session,
      reset_password_db)
from services.vector_store_db import index_document_to_chroma, delete_doc_from_chroma
from services.auth import decode_token, hash_password, create_access_token,verify_password
import os
import uuid
import logging
import shutil
from typing import Dict, List
from PIL import Image
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata"

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Document formats
allowed_extensions = [
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.txt',
    '.jpeg', '.jpg', '.png',
    '.html'
]


@app.get("/chat-history")
def chat_history(user_id:int):
    return get_user_chat_history(user_id)

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):

    session_id = query_input.session_id  or str(uuid.uuid4())
    print("Session ID: ", session_id)
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    chat_history = get_chat_history(session_id,query_input.user_id)
    rag_chain = get_rag_chain(query_input.model.value)
    
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id,query_input.user_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, user_id = query_input.user_id, model=query_input.model)

@app.delete("/delete-chat-history")
def delete_chat_history(user_id:int,session_id:str):
    return delete_chat_session(user_id,session_id)

@app.post("/upload-doc")
def upload_and_index_document(user_id: int, file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}"
        )

    temp_file_path = f"temp_{file.filename}"
    ocr_text_file = f"image_{os.path.splitext(file.filename)[0]}.txt"  # Save .txt file in a temp directory

    try:
        # Save the uploaded image/document file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image files
        if file_extension in ['.jpeg', '.jpg', '.png']:
            # Extract text and save it to a .txt file
            extracted_data = process_image(temp_file_path)
            ocr_text = extracted_data["text"]
            print("OCR Text: ", ocr_text)

            with open(ocr_text_file, "w") as f:
                f.write(ocr_text)

            # Insert into the database and index
            file_id = insert_document_record(file.filename, user_id)
            print("before sucess")
            success = index_document_to_chroma(ocr_text_file, file_id, user_id)

            if not success:
                delete_document_record(file_id, user_id)
                raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")

        # Process document files
        elif file_extension in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.txt', '.html']:
            extracted_data = process_document(temp_file_path, user_id)

        else:
            extracted_data = {"message": "Unsupported but recognized file type. Processing not implemented."}

        return {"message": f"File {file.filename} processed successfully.", "data": extracted_data}

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(ocr_text_file):
            os.remove(ocr_text_file)



@app.get("/list-docs", response_model=List[DocumentInfo])
def list_documents(user_id:int):
    return get_all_documents(user_id)


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id,request.user_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id,request.user_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}



@app.post("/register")
def register(user:UserRegister):
    hashed_password = hash_password(user.password)

    insert_user(user.email, hashed_password)
    response= get_user_by_email(user.email)
    token = create_access_token(data={"sub": response["id"]})
    return {"access_token": token, "token_type": "bearer", "user_id": response["id"]}

@app.post("/login")
def login(userlogin:UserRegister):
    user = get_user_by_email(userlogin.email)
    if not user or not verify_password(userlogin.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": user["id"]})
    response = {"access_token": token, "token_type": "bearer", "user_id": user["id"]}
    return response




def process_image(image_path: str) -> Dict:
    try:
        # Load the image using Pillow
        img = Image.open(image_path)
        custom_config = r'-l deu --psm 6 -c preserve_interword_spaces=1'
        # Extract text using OCR
        ocr_text = pytesseract.image_to_string(img, config=custom_config)

        # Extract image metadata
        metadata = {
            "format": img.format,
            "size": img.size,  # (width, height)
            "mode": img.mode
        }

        return {
            "text": ocr_text,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")



def process_document(file_path: str, user_id: int) -> Dict:
    file_id = insert_document_record(file_path, user_id)
    print("before the chroma db", file_id)
    success = index_document_to_chroma(file_path, file_id, user_id)

    if success:
        return {"file_id": file_id, "status": "indexed"}
    else:
        delete_document_record(file_id, user_id)
        raise HTTPException(status_code=500, detail=f"Failed to index {file_path}.")




@app.post("/reset")
async def reset_password(request: UserRegister):
    user = get_user_by_email(request.email)
    if user:
        hashed_password = hash_password(request.password)
            
        return reset_password_db(request.email, hashed_password)
    else:
        raise HTTPException(status_code=404, detail="User not found")
    




@app.get("/sources/{filename}")
async def get_source(filename: str):
    """Serve source files (PDF or HTML) with appropriate content type.

    Args:
        filename: Name of the file to serve

    Returns:
        FileResponse or HTMLResponse depending on file type

    Raises:
        HTTPException: If file not found or invalid type
    """
    try:
        # Assuming files are stored in a 'data' directory relative to the backend
        base_path = os.path.abspath("data")
        file_path = os.path.normpath(os.path.join(base_path, filename))

        if not file_path.startswith(base_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Determine content type from file extension
        content_type, _ = mimetypes.guess_type(file_path)

        if content_type == "application/pdf":
            return FileResponse(
                file_path,
                media_type="application/pdf",
                filename=filename
            )
        elif content_type in ["text/html", "text/plain"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type"
            )

    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

