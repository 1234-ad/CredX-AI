# file: app.py

import streamlit as st
import json # Used to parse user's JSON input
from matching_engine import Recommender
from resume_parser import ResumeParser
import os

# The following Flask-related imports and variables have been REMOVED as they cause the crash:
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)

# --- Configuration and Initialization ---

DATA_FILE_PATH = os.path.join('data', 'jobs.csv')

# Use st.secrets for the API Key in the cloud environment.
# You must set 'API_KEY' in your app's secrets.toml file.
try:
    API_KEY = st.secrets["API_KEY"]
except KeyError:
    API_KEY = "YOUR_API_KEY_HERE"

# Initialize Google Generative AI (same logic as yours)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

if API_KEY != "YOUR_API_KEY_HERE" and genai:
    try:
        genai.configure(api_key=API_KEY)
        st.success("Google Generative AI configured successfully.")
    except Exception as e:
        st.warning(f"Error configuring Google AI: {e}")
else:
    st.warning("WARNING: API key not set in Streamlit secrets or 'google-generativeai' is not installed.")


# Use st.cache_resource to initialize heavy objects only once
@st.cache_resource
def initialize_engine():
    st.info("Initializing the recommendation engine...")
    recommender = Recommender(DATA_FILE_PATH, api_key=API_KEY)
    resume_parser = ResumeParser(api_key=API_KEY)
    st.info("Recommendation engine initialized successfully.")
    return recommender, resume_parser

recommender, resume_parser = initialize_engine()


# --- Streamlit Front-End Interface (Replacing Flask Routes) ---

st.title("CredX AI: Job Recommendation and Resume Parsing")

# Create two tabs for the two functionalities
tab_recommend, tab_parse = st.tabs(["Job Recommendation", "Resume Parsing"])


# --- Job Recommendation Tab (Replaces @app.route('/recommend')) ---

with tab_recommend:
    st.header("Job Recommendation Engine")
    st.write("Enter your job payload as a JSON string to receive recommendations.")
    
    json_input = st.text_area(
        "JSON Payload (e.g., {'user_skills': ['python', 'ai'], 'years_exp': 5})",
        height=200,
        key="recommend_json"
    )

    if st.button("Get Recommendations"):
        if not json_input:
            st.error("Invalid JSON payload.")
            st.stop()
            
        try:
            # Replaces Flask's request.get_json()
            request_data = json.loads(json_input)
            
            # Call the core logic function
            recommendations = recommender.get_recommendations(request_data)
            
            # Display results (replaces return jsonify(recommendations))
            st.subheader("Recommended Jobs")
            st.json(recommendations)
            
        except json.JSONDecodeError:
            st.error("Error: Invalid JSON format provided.")
        except Exception as e:
            st.error("An internal server error occurred.")
            st.exception(e)


# --- Resume Parsing Tab (Replaces @app.route('/parse_resume')) ---

with tab_parse:
    st.header("Resume Parsing")
    
    # Streamlit's file uploader replaces Flask's request.files['resume']
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=['pdf'])
    
    if st.button("Parse Resume"):
        
        if uploaded_file is None:
             st.error("No resume file provided or selected.")
             st.stop()
             
        try:
            # The Streamlit file object is passed directly to the parser
            extracted_data = resume_parser.parse(uploaded_file)
            
            # Display results (replaces return jsonify(extracted_data))
            st.subheader("Extracted Data")
            st.json(extracted_data)
            
        except Exception as e:
            st.error("Failed to parse the resume.")
            st.exception(e)
