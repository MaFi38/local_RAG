import streamlit as st
import os
from components.api_utils import (
    upload_document, 
    list_documents, 
    delete_document,
    user_chat_history, 
    delete_session)

def display_documents():
    user_id = st.session_state.user_id
   

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=[
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.txt',
    '.jpeg', '.jpg', '.png',
    '.html'
])
    if uploaded_file and st.sidebar.button("Upload"):
        with st.spinner("Uploading..."):
            user_id = st.session_state.user_id

            upload_response = upload_document(uploaded_file,user_id)
            if upload_response:
                st.sidebar.success(f"File uploaded successfully.")
                st.session_state.documents = list_documents(user_id)

    # List and delete documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        st.session_state.documents = list_documents(user_id)

    if "documents" in st.session_state and st.session_state.documents:
        for doc in st.session_state.documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']})")

        selected_file_id = st.sidebar.selectbox(
            "Select a document to delete",
            options=[doc['id'] for doc in st.session_state.documents],
        )
        if st.sidebar.button("Delete Selected Document"):
            delete_response = delete_document(selected_file_id,user_id)
            if delete_response:
                st.sidebar.success(f"Document deleted successfully.")
                st.session_state.documents = list_documents(user_id)
