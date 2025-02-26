import streamlit as st
import os
from components.api_utils import (
    upload_document, 
    list_documents, 
    delete_document,
    user_chat_history, 
    delete_session)
from datetime import datetime as dt




def display_sidebar(create_new_session, switch_session):
    if st.sidebar.button("New Chat"):
        create_new_session()

    st.session_state.sessions = user_chat_history(st.session_state.user_id)
    
    if st.session_state.current_session:
        current_session_id = st.session_state.current_session
        if current_session_id not in st.session_state.sessions:
            st.session_state.sessions[current_session_id] = {
                "model": "llama3.1",
                "timestamp": dt.now(), 
                "queries": [],
            }

        model_options = ["llama3.2","llama3.1","gpt-4o", "gpt-4o-mini"]
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=model_options,
            index=model_options.index(st.session_state.sessions[current_session_id]["model"]),
        )
        st.session_state.sessions[current_session_id]["model"] = selected_model


    sorted_sessions = sorted(
        st.session_state.sessions.items(),
        key=lambda x: dt.strptime(x[1]["timestamp"], "%Y-%m-%d %H:%M:%S") if isinstance(x[1]["timestamp"], str) else x[1]["timestamp"], 
        reverse=True  
    )

    for session_id, session_data in sorted_sessions:
        first_chat_title = session_data["queries"][0]["query"] if session_data["queries"] else "Untitled Session"
        first_chat_title = str(first_chat_title[:30]) + "..." if len(first_chat_title) > 30 else str(first_chat_title)
        
        col1, col2 = st.sidebar.columns([4, 1]) 
        with col1:
            # Add a unique key for each session title button
            if st.button(f"{first_chat_title}", key=f"switch_{session_id}"):  
                switch_session(session_id)

        with col2:
            # Add a unique key for the delete button
            delete_button = st.button("Delete", key=f"delete_{session_id}")
            if delete_button: 
                del st.session_state.sessions[session_id] 
                user_id = st.session_state.user_id
                delete_session(session_id, user_id)

                if st.session_state.current_session == session_id:
                    st.session_state.current_session = None 
                st.session_state["refresh_key"] = st.session_state.get("refresh_key", 0) + 1

         