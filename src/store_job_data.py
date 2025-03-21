"""
store_job_data.py

This script processes job data from a CSV file, generates embeddings using a Hugging Face model, 
and stores them in a FAISS index for efficient retrieval.

Main functionalities:
1. Loads job data from a CSV file.
2. Uses a Hugging Face transformer model to generate text embeddings.
3. Embeddings are generated using **full job details**, now including **Job Title**.
4. Stores the embeddings in a FAISS vector index.
5. Saves job details in a JSON file for reference.

Dependencies:
- Python 3.8+
- `faiss`, `numpy`, `pandas`, `json`
- `torch`, `transformers`
- `tqdm` for progress tracking
"""

import faiss
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Load CSV file
CSV_FILE = "../scraped_jobs.csv"  # Ensure this file exists
print("📂 Loading job data...")
df = pd.read_csv(CSV_FILE)

# Load Hugging Face model for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("📥 Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    """
    Converts input text into an embedding vector using a transformer model.

    :param text: The text to convert into an embedding.
    :return: NumPy array containing the embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use CLS token embedding

# Process jobs and generate embeddings
embeddings = []
job_data = []
print("🔄 Processing job listings...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    # ✅ Now **Job Title is included** in full job details for embedding
    full_text = (
        f"Job Title: {row['Job Title']}, Company: {row['Company Name']}, "
        f"Skills: {row['Required Skills']}, Experience: {row['Experience']} years, "
        f"Salary: {row['Salary']}, Location: {row['Location']}, More Info: {row['More Info']}"
    )
    
    job_object = {
        "jobTitle": row["Job Title"],
        "company": row["Company Name"],
        "requiredSkills": row["Required Skills"],
        "experience": str(row["Experience"]),
        "salary": str(row["Salary"]),
        "location": row["Location"],
        "moreInfo": row["More Info"]
    }

    job_data.append(job_object)  # Save full job details
    emb = get_embedding(full_text)  # Get embedding of full job details including Job Title
    embeddings.append(emb)

# Convert list to FAISS format
embeddings = np.vstack(embeddings).astype("float32")

# Create and save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "../Data_files/job_index.faiss")

# Save job details to JSON
with open("../Data_files/job_data.json", "w") as f:
    json.dump(job_data, f)

print("✅ FAISS index and job details saved successfully!")
