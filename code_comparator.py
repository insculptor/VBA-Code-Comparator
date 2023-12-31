import streamlit as st
import os
import zipfile
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import numpy as np

# Function to extract VBA code from files
def extract_vba_code(zip_file_path):
    vba_code = []
    extracted_files = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        for file_info in zip_file.infolist():
            if file_info.filename.endswith(".vba"):
                with zip_file.open(file_info) as vba_file:
                    vba_code.append(vba_file.read().decode("utf-8"))
                    extracted_files.append(file_info.filename)
    return vba_code, extracted_files

# Function to calculate similarity
def calculate_similarity(vba_code):
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words=list(stop_words))
    vectors = vectorizer.fit_transform(vba_code)
    similarity_matrix = cosine_similarity(vectors, vectors)
    return similarity_matrix

# Function to generate file summary statistics
def file_summary_statistics(vba_code, extracted_files):
    summary_data = []
    for idx, code in enumerate(vba_code):
        lines = code.split('\n')
        sub_count = code.count("Sub ")
        if_count = code.count("If ")
        for_count = code.count("For ")
        summary_data.append([extracted_files[idx], len(lines), sub_count, if_count, for_count])
    return pd.DataFrame(summary_data, columns=["File Name", "Line of Code", "Sub Count", "If Count", "For Loop Count"])

# Function to generate file comparison
def file_comparison(vba_code, extracted_files):
    similarity_matrix = calculate_similarity(vba_code)
    comparison_data = []

    for i in range(len(vba_code)):
        for j in range(i + 1, len(vba_code)):
            similarity_score = similarity_matrix[i][j]
            matching_lines = sum(1 for a, b in zip(vba_code[i].splitlines(), vba_code[j].splitlines()) if a == b)
            non_matching_lines = len(vba_code[i].splitlines()) - matching_lines
            percentage_matching_lines = (matching_lines / len(vba_code[i].splitlines())) * 100
            comparison_data.append([extracted_files[i], extracted_files[j], len(vba_code[i].split()), similarity_score, matching_lines, non_matching_lines, percentage_matching_lines])

    comparison_df = pd.DataFrame(comparison_data, columns=["File 1 Name", "File 2 Name", "Token Count", "Similarity Score", "Matching Lines", "Non-Matching Lines", "% Matching Lines"])
    comparison_df = comparison_df.sort_values(by="Similarity Score", ascending=False)
    return comparison_df

# Function to create scatter plot
def create_scatter_plot(vba_code):
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words=list(stop_words))
    vectors = vectorizer.fit_transform(vba_code)
    
    # Perform KMeans clustering
    num_clusters = 3  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors.toarray())
    
    # Reduce dimensions for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(vectors.toarray())
    
    df = pd.DataFrame(reduced_features, columns=["Dimension 1", "Dimension 2"])
    df["Cluster"] = cluster_labels
    df["File Name"] = extracted_files
    
    # Create a scatter plot
    fig = px.scatter(df, x="Dimension 1", y="Dimension 2", color="Cluster", text="File Name")
    fig.update_traces(textposition='top center')
    return fig

# Streamlit UI
st.title("VBA Code Comparison and Clustering")

# File upload and processing
uploaded_file = st.file_uploader("Upload a zip file containing VBA files", type=["zip"])
if uploaded_file is not None:
    with st.spinner("Processing..."):
        # Extract VBA code and file names
        vba_code, extracted_files = extract_vba_code(uploaded_file)

        # Show extracted file names
        st.subheader("Extracted Files")
        st.write(extracted_files)

        # Show file summary statistics
        st.subheader("File Summary Statistics")
        file_summary_df = file_summary_statistics(vba_code, extracted_files)
        st.dataframe(file_summary_df)

        # Show file comparison
        st.subheader("File Comparison")
        file_comparison_df = file_comparison(vba_code, extracted_files)
        st.dataframe(file_comparison_df)

        # Create and display the scatter plot
        st.subheader("File Clustering Scatter Plot")
        scatter_plot = create_scatter_plot(vba_code)
        st.plotly_chart(scatter_plot)
