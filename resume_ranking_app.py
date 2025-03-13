import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip() if text.strip() else "No text found in PDF"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    if not resumes:
        return []
    
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    # Convert scores to whole numbers (0-100 scale)
    scores = (cosine_similarities * 100).astype(int)
    
    return scores

# Streamlit app with enhanced UI for HR
st.set_page_config(page_title="AI Resume Ranking System", page_icon="📄", layout="wide")

# Custom CSS for dark mode, animations, and UI improvements
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .big-title {
        font-size: 45px;
        font-family: Merriweather;
        font-weight: bold;
        color: #0a0101;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 2s ease-in-out;
    }
    .sub-title {
        font-size: 22px;
        color: #807a7a;
        text-align: center;
        margin-bottom: 40px;
        animation: fadeIn 3s ease-in-out;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-size: 20px;
        padding: 12px 26px;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #17A74B;
    }
    .dataframe-table {
        border-radius: 12px;
        box-shadow: 0px 4px 8px rgba(255, 255, 255, 0.2);
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# Title with animation and dark mode
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='big-title'>JobFit : AI Resume Ranking System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Helping HR professionals rank resumes quickly and accurately!</div>", unsafe_allow_html=True)

with col2:
    st.image("https://github.com/vaishnavi80800/AI-Resume-Ranking-System/blob/main/Resume%20image.jpg?raw=true", width=150)

# Job description input with enhanced UI
st.header("📌 Job Description")
job_description = st.text_area("Enter the job description", placeholder="E.g., Looking for a Python developer with AI/ML experience...")

# File uploader with modern design
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("📎 Upload PDF resumes here", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("🏆 Ranking Resumes")
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    
    if scores is None or len(scores) == 0:
        st.warning("⚠ No valid resumes found. Please check the uploaded files.")
    else:
        # Create a DataFrame with ranking
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        results["Rank"] = range(1, len(results) + 1)  # Assign rank based on descending order
        
        # Display top results with a visually appealing table
        st.markdown("### 📋 Top Ranked Resumes")
        st.dataframe(results.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#1DB954'), ('color', 'white'), ('font-size', '16px')]}]))

        st.success("🎉 Ranking completed! Check the top candidates above.")
        
        # Download button for ranked results
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv",
        )
else:
    st.info("🔹 Please upload resumes and enter a job description to proceed.")
