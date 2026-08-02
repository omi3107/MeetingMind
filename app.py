"""
app.py — MeetingMind AI Dashboard
Streamlit UI matching the meetingmind.html wireframe layout.
Uses ML classification + Gemini/Groq AI framing pipeline.
"""
import json
import logging
import sys
import io
import os
import uuid
import time
import html as html_mod
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
try:
    import markdown
except ImportError:
    markdown = None
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import threading
import uvicorn
import time
import os
from backend.api.analysis_server import app as fastapi_app

# Inject Streamlit secrets into environment variables for the backend to use
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ[key] = str(value)
except Exception:
    pass # No secrets configured or running locally

@st.cache_resource
def start_fastapi():
    def run_server():
        # Run uvicorn on the background thread
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8502)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Optional: Wait a tiny bit to let the server spin up
    time.sleep(1)
    return thread

# Start the FastAPI server silently in the background (runs only once per deployment)
start_fastapi()

st.set_page_config(page_title='MeetingMind', page_icon='🧠', layout='wide', initial_sidebar_state='collapsed')

# Keep Streamlit chrome out of the way so the embedded UI feels like the original page.
st.markdown(
    """
    <style>
      #MainMenu, footer, header { visibility: hidden; display: none !important; }
      header[data-testid="stHeader"] { display: none !important; }
      .stApp { background: #0c0e14; }
      [data-testid="stSidebar"] { display: none; }
      .block-container { 
          padding-top: 0 !important; 
          padding-bottom: 0 !important; 
          padding-left: 0 !important; 
          padding-right: 0 !important; 
          margin-top: 0 !important;
          max-width: 100% !important; 
      }
      div[data-testid="stVerticalBlock"] > div:first-child {
          padding-top: 0 !important;
      }
      /* Hide Streamlit iframe borders */
      iframe { border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

import requests

frontend_dir = Path(__file__).parent / "frontend"
if not frontend_dir.exists():
    st.error(f'Could not find frontend directory: {frontend_dir}')
    st.stop()

# Declare the Streamlit component
my_component = components.declare_component("meetingmind", path=str(frontend_dir))

# Initialize last_response in session state
if 'last_response' not in st.session_state:
    st.session_state.last_response = None

# Render the component, passing down any new response
component_value = my_component(key="meetingmind_ui", response=st.session_state.last_response, height=650)

if component_value and "id" in component_value:
    req_id = component_value["id"]
    
    # Process only if this is a new request
    if not st.session_state.last_response or st.session_state.last_response.get("id") != req_id:
        path = component_value.get("url")
        method = component_value.get("method", "GET")
        body = component_value.get("body")
        
        try:
            url = f"http://localhost:8502{path}"
            if method == "POST":
                if path == "/api/extract_text" and body and "file_base64" in body:
                    import base64
                    file_bytes = base64.b64decode(body["file_base64"])
                    files = {"file": (body["filename"], file_bytes)}
                    res = requests.post(url, files=files)
                else:
                    res = requests.post(url, json=body)
            else:
                res = requests.get(url)
                
            res.raise_for_status()
            response_data = res.json()
            
            st.session_state.last_response = {
                "id": req_id,
                "data": response_data
            }
        except Exception as e:
            st.session_state.last_response = {
                "id": req_id,
                "error": str(e)
            }
        
        # Rerun to pass the response back into the component
        st.rerun()


