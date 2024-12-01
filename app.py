import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import warnings
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import PyPDF2
import io

from PIL import Image

# Define HTML/CSS style with background image and white font color for the text
html_style = """
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url('https://images.unsplash.com/photo-1642355008521-236f1d29d0a8?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        background-position: center;
        height: 100%;
        padding-top: 50px; /* Adds some space for the header */
    }

    /* Body style adjustments */
    body {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        background-color: rgba(0, 0, 0, 0.5); /* Transparent black overlay */
        color: #ffffff; /* White font color for text */
    }

    /* Container styling */
    .container {
        color: #ffffff;
        opacity: 0.9;
        padding: 30px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255); /* White with transparency */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    ol,ul {
        color: #ffffff;
    }
    h1 {
        color: #ffffff; /* Gold for the title */
        font-size: 40px;
        text-align: center;
    }

    h2, h3 {
        color: #ffffff;
        font-size: 28px;
        margin-top: 20px;
    }

    p {
        color: #ffffff;
        font-size: 18px;
        line-height: 1.5;
        margin-bottom: 15px;
    }

    .file-upload-section {
        color: #ffffff;
        border: 1px dashed #ffffff; 
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .file-upload-section input {
        color: #ffffff;
    }

    .prediction-result {
        background-color: rgba(0, 128, 0);
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        font-size: 18px;
        margin-top: 15px;
    }

    .metric-box {
        padding: 20px;
        border: 1px solid #ffffff;
        background-color: rgba(255, 255, 255);
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
"""

# Render HTML/CSS style
st.markdown(html_style, unsafe_allow_html=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit setup
st.title("Resume ATS Score Checker")

# Introduction for the user
st.write("""
### Welcome to the Resume ATS Score Checker App!

Upload your resume in pdf format for which you want to check the ATS score.

Start by uploading your resumes!
""")

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_pdf):
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    text = ''
    for page in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    return text

# File path to the CSV file (automatically accessed)
csv_file_path = r'UpdatedResumeDataSet.csv'

# Load the CSV file for training
data = pd.read_csv(csv_file_path, encoding='utf-8')

# Text cleaning function
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

data['cleaned_resume'] = data['Resume'].apply(lambda x: cleanResume(x))

# Label Encoding for categories
le = LabelEncoder()
data['Category'] = le.fit_transform(data['Category'])

# Prepare data for training
requiredText = data['cleaned_resume'].values
requiredTarget = data['Category'].values
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=42, test_size=0.2, shuffle=True, stratify=requiredTarget)

# Train a classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)



# File uploader for PDFs (User input)
uploaded_pdf = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

if uploaded_pdf:
    # Extract text from each uploaded PDF file
    for pdf in uploaded_pdf:
        pdf_text = extract_text_from_pdf(pdf)
        
        # Preprocess and predict
        processed_text = cleanResume(pdf_text)
        features = word_vectorizer.transform([processed_text])
        prediction = clf.predict(features)
        predicted_category = le.inverse_transform(prediction)
        
        st.subheader(f"Prediction for {pdf.name}")
        st.write(f"The uploaded resume is best suited for the following job category: **{predicted_category[0]}**")
else:
    st.write("Please upload a PDF resume for prediction.")

# Display only final performance metrics
st.subheader("Model Performance")
st.write(f'Accuracy on training set: {clf.score(X_train, y_train):.2f}')
st.write(f'Accuracy on test set: {clf.score(X_test, y_test):.2f}')
