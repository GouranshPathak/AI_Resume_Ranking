import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS for better visibility and UI improvements
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stApp {
            background-color: #ffffff;
            color: #333;
        }
        h1 {
            color: #007BFF;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
        }
        .stButton>button {
            background: #007BFF;
            color: white;
            padding: 12px 20px;
            font-size: 18px;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: #0056b3;
        }
        .stDataFrame tbody tr:hover {
            background-color: #f1f1f1 !important;
        }
        /* Custom sidebar title color */
        .sidebar-title {
            color: #FF5733 !important;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        /* Improved text visibility */
        .highlight-box {
            background-color: #eaf6ff;
            padding: 12px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            color: #0056b3;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)

st.sidebar.markdown('<h1 class="sidebar-title">📂 Resume Ranking System</h1>', unsafe_allow_html=True)
st.sidebar.info("Upload resumes and enter job descriptions to rank them based on relevance.")

uploaded_files = st.sidebar.file_uploader("📤 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
st.sidebar.write("---")
job_description = st.sidebar.text_area("📝 Job Description", "")
st.sidebar.write("---")
rank_button = st.sidebar.button("📊 Rank Resumes")

# Main UI
st.title("🚀 AI-Powered Resume Ranking")
st.markdown("### 🔍 Find the best match for your job opening!")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    return " ".join([page.extract_text() or "" for page in pdf.pages]).strip()

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_desc_vector], resume_vectors).flatten()

# Rank resumes when the button is clicked
if rank_button:
    if uploaded_files and job_description.strip():
        st.header("📊 Ranked Resumes")
        
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]
        
        # Progress bar for better UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            status_text.text(f"🔄 Processing... {i+1}%")
        
        scores = rank_resumes(job_description, resumes)
        results_df = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Match Score (%)": (scores * 100).round(2)
        }).sort_values(by="Match Score (%)", ascending=False)
        
        top_match = results_df.iloc[0]["Resume"]
        
        st.markdown(f'<div class="highlight-box">✅ Ranking Complete! 🎯 Top Match: <br> <span style="font-size:20px; font-weight:bold;">{top_match}</span></div>', unsafe_allow_html=True)
        
        st.dataframe(results_df.style.format({"Match Score (%)": "{:.2f}"}))
    else:
        st.warning("⚠️ Please upload resumes and enter a job description first!")
