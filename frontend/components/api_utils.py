import requests
import streamlit as st

def get_api_response(question, session_id, model,user_id):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"question": question, "model": model, "user_id": user_id}
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post("http://127.0.0.1:8000/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def upload_document(file,user_id):
    print(f"user_id: {user_id}")
    print(f"Uploaded file: {file.name}")
    url = f"http://127.0.0.1:8000/upload-doc"
    try:
        files = {"file": (file.name, file, file.type)}
        params = {"user_id": user_id} 
        response = requests.post(url,params=params,files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents(user_id):
    try:
        params = {"user_id": user_id}
        response = requests.get("http://127.0.0.1:8000/list-docs",params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id,user_id):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"file_id": file_id, "user_id": user_id}

    try:
        response = requests.post("http://127.0.0.1:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None

def user_chat_history(user_id):
    try:
        params = {"user_id": user_id}
        response = requests.get("http://127.0.0.1:8000/chat-history",params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch chat history. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the chat history: {str(e)}")
        return []



def authenticate_user_via_api(email, password):
    """Authenticate user by making an API call."""
    url = "http://127.0.0.1:8000/login"
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"email": email, "password": password}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()  # Return the result for successful login
        else:
            return {"error": f"Login failed: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"An error occurred during login: {str(e)}"}
    

def register_user_via_api(email, password):
    """Register user by making an API call."""
    url = "http://127.0.0.1:8000/register"
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"email": email, "password": password}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()  # Return the result for successful registration
        else:
            return {"error": f"Registration failed: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"An error occurred during registration: {str(e)}"}


def delete_session(session_id,user_id):
    # headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    params = {"session_id": session_id, "user_id": user_id}

    try:
        response = requests.delete("http://127.0.0.1:8000/delete-chat-history",params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete session. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the session: {str(e)}")
        return None



def reset_password_api(email, new_password):
    """Send password reset request to API."""
    url = "http://127.0.0.1:8000/reset"
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"email": email, "password": new_password}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()  # Return success message
        else:
            return {"error": f"Password reset failed: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"An error occurred while resetting password: {str(e)}"}