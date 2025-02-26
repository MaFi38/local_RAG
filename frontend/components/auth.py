import streamlit as st
import hashlib
from components.chat_interface import display_chat_interface
from components.sidebar import display_sidebar
from components.utils import create_new_session, switch_session  
from components.api_utils import authenticate_user_via_api, register_user_via_api,reset_password_api
from components.api_utils import user_chat_history


def login_page():
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = authenticate_user_via_api(email, password)
        if "error" in result:
            st.error(result["error"])  # Show error if login fails
        else:
            st.success("Login successful!")
            st.session_state.authenticated = True
            st.session_state.access_token = result["access_token"]
            st.session_state.token_type = result["token_type"]
            st.session_state.user_id = result["user_id"]
            create_new_session()  # Initialize a new session on successful login
          

def signup_page():
    st.title("Sign Up")

    email = st.text_input("Enter Email")
    password = st.text_input("Enter Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters long")
        else:
            result = register_user_via_api(email, password)
            if "error" in result:
                
                st.error("Username already exists. Please choose a different one.")
            else:
                st.success("Account created successfully!")
                st.session_state.authenticated = True
                st.session_state.access_token = result["access_token"]
                st.session_state.token_type = result["token_type"]
                st.session_state.user_id = result["user_id"]
                create_new_session()  
                st.session_state.sessions = user_chat_history(st.session_state.user_id)


def forgot_password_page():
    st.title("Reset Password")

    email = st.text_input("Enter your registered email")
    new_password = st.text_input("Enter new password", type="password")
    confirm_password = st.text_input("Confirm new password", type="password")

    if st.button("Reset Password"):
        if new_password != confirm_password:
            st.error("Passwords do not match")
        elif len(new_password) < 6:
            st.error("Password must be at least 6 characters long")
        else:
            result = reset_password_api(email, new_password)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Password reset successfully! Please login with your new password.")
                st.session_state.page = "login"

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "page" not in st.session_state:
        st.session_state.page = "login"

    if st.session_state.authenticated:
        st.session_state.sessions = user_chat_history(st.session_state.user_id)
        display_sidebar(create_new_session, switch_session)
        display_chat_interface()

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
    else:
        if st.session_state.page == "login":
            login_page()
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Go to Sign Up"):
                    st.session_state.page = "signup"
            with col2:
                if st.button("Forgot Password?"):
                    st.session_state.page = "forgot_password"
        elif st.session_state.page == "signup":
            signup_page()
            if st.button("Go to Login"):
                st.session_state.page = "login"

        elif st.session_state.page == "forgot_password":
            forgot_password_page()
            if st.button("Go to Login"):
                st.session_state.page = "login"
