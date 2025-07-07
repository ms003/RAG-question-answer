
# pip install PyPDF2 langchain transformers faiss-cpu

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch
import os

# get all PDF files from the 'data' directory
def get_pdf_files(directory):
    pdf_files = [os.path.join(directory, file)\
                 for file in os.listdir(directory) if file.endswith(".pdf")]
    return pdf_files
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# chunk-dataperparagraph
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# vectorize the datainto a db
def vectorize_chunks(chunks, tokenizer, model):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean pooling of the last hidden state as the embedding
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

# Step 5: Store vectors in FAISS
def store_vectors(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

# Main function
def build_vector_database(pdf_files):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Load tokenizer
    model = AutoModel.from_pretrained("distilbert-base-uncased")  # Load pre-trained model
    all_vectors = []
    for pdf_file in pdf_files:
        text = load_pdf(pdf_file)
        chunks = chunk_text(text)
        vectors = vectorize_chunks(chunks, tokenizer, model)
        all_vectors.append(vectors)
    all_vectors = np.vstack(all_vectors)  # Combine all vectors
    index = store_vectors(all_vectors)
    return index

# Example usage

data_directory = "./data"
pdf_files = get_pdf_files(data_directory)
# Replace with your PDF file paths
vector_db = build_vector_database(pdf_files)
print("Vector database built successfully!")

