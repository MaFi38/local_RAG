import streamlit as st
from components.api_utils import get_api_response

def display_chat_interface():
    if st.session_state.current_session is None:
        st.write("Create or select a session to start chatting.")
        return

    current_session_id = st.session_state.current_session
    session_data = st.session_state.sessions[current_session_id]
    
    # Ensure 'queries' list exists, if not initialize it as empty
    if "queries" not in session_data:
        session_data["queries"] = []

    # Display chat history (user and assistant messages)
    for message in session_data["queries"]:
        role_q = "user" 
        content_q = message["query"] 
        
        role_a = "assistant"
        content_a = message["response"]
        
        with st.chat_message(role_q):
            st.markdown(content_q)

        with st.chat_message(role_a):
            st.markdown(content_a)

    # Handle new user input
    if prompt := st.chat_input("Query:"):
        session_data["queries"].append({"query": prompt, "response": ""})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get API response
        with st.spinner("Generating response..."):
            response = get_api_response(prompt, current_session_id, session_data["model"], st.session_state.user_id)

            if response and "answer" in response:
                session_data["queries"][-1]["response"] = response["answer"]
                
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])

            else:
                st.error("Failed to get a response from the API. Please try again.")