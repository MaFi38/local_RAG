from datetime import datetime as dt
import streamlit as st
import uuid
from components.sidebar import display_sidebar
from components.chat_interface import display_chat_interface
from components.auth import main
from components.sidebar import display_sidebar
from components.chat_interface import display_chat_interface
from components.utils import create_new_session, switch_session
from components.api_utils import user_chat_history
from components.document import display_documents

import base64

# @st.cache_data
# def get_image_as_base64(image_path):
#     with open(image_path, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# Update the image path to your SVG file
# image_path = "static/POC_Copilot.svg"
# img_base64 = get_image_as_base64(image_path)

# st.markdown(
#     """
#     <style>
#     /* Container for header elements */
#     .fixed-header {
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         color: white; /* or any color that matches your design */
#         background-color: Black; /* or any color that matches your design */
#         z-index: 1000;
#         text-align: center;
#         padding: 10px 0;
#         /* border-bottom: 1px solid #ddd; */
#     }
#     .fixed-header img {
#         display: block;
#         margin-left: auto;
#         margin-right: auto;
#     }
#     /* Padding for the main content so it doesn't get hidden behind the fixed header */
#     .content {
#         margin-top: 250px;  /* Adjust this value based on your header height */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Update the MIME type from image/jpeg to image/svg+xml
# header_html = f"""
# <div class="fixed-header">
#     <img src="data:image/svg+xml;base64,{img_base64}" width="600" height ="300" alt="Logo">
#     <h1>PoC Copilot</h1>
# </div>
# """

# st.markdown(header_html, unsafe_allow_html=True)
st.markdown('<div class="content">', unsafe_allow_html=True)

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    main()
else:
    display_sidebar(create_new_session, switch_session)
    display_chat_interface()
    display_documents()
