import uuid
from datetime import datetime as dt
import streamlit as st

def create_new_session():
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "messages": [],
        "model": "gpt-4o",
        "timestamp": dt.now(),
    }
    st.session_state.current_session = session_id

def switch_session(session_id):
    st.session_state.current_session = session_id
