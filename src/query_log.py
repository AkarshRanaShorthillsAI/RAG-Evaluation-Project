"""
query_faiss_batch.py

Enhanced job search pipeline with batch query processing capabilities.
Processes multiple queries from a JSON file with 30-second delays between requests.
"""

import os
import faiss
import json
import numpy as np
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Google Gemini API key not found! Set GEMINI_API_KEY.")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# Base directory (assumes script is inside `src/`)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
LOG_FILE = os.path.join(BASE_DIR, "Data_files", "query_logs.json")
PREDICTIONS_FILE = os.path.join(BASE_DIR, "Data_files", "predictions.json")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "Data_files", "job_index.faiss")
JOB_DATA_FILE = os.path.join(BASE_DIR, "Data_files", "job_data.json")

def log_interaction(query, retrieved_jobs, ai_response):
    """Logs user queries, retrieved job results, and AI-generated responses."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "retrieved_jobs": retrieved_jobs,
        "ai_response": ai_response,
    }

    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

    print("✅ Interaction logged!")

def save_predictions(query, selected_links):
    """Saves the most relevant job links into `predictions.json`."""
    predictions = []
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    predictions.append({"query": query, "relevant_jobs": selected_links})

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    print("✅ Predictions logged successfully!")

def load_faiss_index():
    """Loads the FAISS index."""
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        print("✅ FAISS index loaded successfully!")
        return index
    except Exception as e:
        raise ValueError(f"❌ Error loading FAISS index: {e}")

def load_job_data():
    """Loads job data from JSON."""
    try:
        with open(JOB_DATA_FILE, "r", encoding="utf-8") as f:
            job_data = json.load(f)
        print(f"✅ Loaded {len(job_data)} job listings.")
        return job_data
    except Exception as e:
        raise ValueError(f"❌ Error loading job data JSON: {e}")

# Load FAISS and job data
index = load_faiss_index()
job_data = load_job_data()

# Load Hugging Face embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)

def get_embedding(text):
    """Converts query text into an embedding vector."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().astype("float32")

def extract_top_5_links(ai_response, matched_jobs):
    """Extracts job links for the top 5 refined jobs from Gemini's response."""
    extracted_links = []
    response_lines = ai_response.split("\n\n")

    for job_text in response_lines:
        for job in matched_jobs:
            if job["company"].lower() in job_text.lower() and job["jobTitle"].lower() in job_text.lower():
                job_link = job.get("moreInfo", "No link available")
                if job_link and job_link not in extracted_links:
                    extracted_links.append(job_link)
                    break  

    while len(extracted_links) < 5:
        extracted_links.append("No valid link found")

    return extracted_links[:5]

def search_jobs(query, k=10, return_matched_jobs=False):
    """
    Searches FAISS for job listings and refines the results using Google Gemini.
    
    Args:
        query (str): User search query
        k (int): Number of job results to retrieve
        return_matched_jobs (bool): Whether to return matched jobs list

    Returns:
        str | tuple: Refined response or (response, matched_jobs) tuple
    """
    query_embedding = get_embedding(query).reshape(1, -1)

    distances, indices = index.search(query_embedding, k)

    matched_jobs = [job_data[idx] for idx in indices[0] if 0 <= idx < len(job_data)]

    if not matched_jobs:
        response = "⚠️ No jobs found. Try a different query."
        log_interaction(query, [], response)
        return (response, []) if return_matched_jobs else response

    job_context = "\n\n".join(
        [
            f"🏢 **Company:** {job['company']}\n"
            f"📌 **Job Title:** {job['jobTitle']}\n"
            f"💼 **Skills:** {job['requiredSkills']}\n"
            f"📅 **Experience:** {job['experience']} years\n"
            f"📍 **Location:** {job['location']}\n"
            f"🔗 **More Info:** {job['moreInfo']}"
            for job in matched_jobs
        ]
    )

    prompt = f"""
    You are an AI job assistant. Based on the given user query, refine the provided job listings 
    and show the exactly **5 most relevant** ones.
    NO matter what show 5 jobs in response , if they are not relevant then try to select the most closest ones , just make sure to take them from context.

    **Rules:**
    - Select the most relevant jobs based **only on the provided listings**.
    - Display jobs that match the required experience level.
    - Do not add or modify information—only filter and reformat.
    - The output should follow this structure:

      "**[Company Name]** is Hiring for **[jobTitle]** role
      They are looking for candidates with **[Required Skills]** and require 
      **[Experience] years** of experience.  
      The job is based in **[Location]**. 
      If you're interested, click below to apply: [More Info]" 

    **User Query:** {query}

    **Job Listings:**

    {job_context}
    """

    try:
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        refined_results = response.text.strip() if hasattr(response, "text") else "No response."
        selected_links = extract_top_5_links(refined_results, matched_jobs)

    except Exception as e:
        refined_results = f"Error in Gemini API: {e}"
        selected_links = []

    # save_predictions(query, selected_links)  # Uncomment if needed
    log_interaction(query, matched_jobs, refined_results)

    return (refined_results, matched_jobs) if return_matched_jobs else refined_results

def process_queries_batch(input_json_path, output_json_path):
    """
    Processes queries from a JSON file with 30-second delays between requests.
    
    Args:
        input_json_path (str): Path to JSON file with array of {"query": "..."} objects
        output_json_path (str): Path to save results with links
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        print(f"✅ Loaded {len(queries)} queries from {input_json_path}")
    except Exception as e:
        print(f"❌ Error loading queries: {e}")
        return

    results = []
    
    for idx, query_obj in enumerate(queries):
        query = query_obj.get("query", "")
        if not query:
            print(f"⚠️ Skipping invalid query object: {query_obj}")
            continue
            
        print(f"\nProcessing query {idx+1}/{len(queries)}: {query}")
        
        try:
            refined_response, matched_jobs = search_jobs(query, return_matched_jobs=True)
            links = extract_top_5_links(refined_response, matched_jobs)
            
            results.append({
                "query": query,
                "relevant_jobs": links,
            })
            
            if idx < len(queries)-1:
                print("⏳ Waiting 15 seconds before next query...")
                time.sleep(15)
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            results.append({
                "query": query,
                "error": str(e),
            })

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Saved {len(results)} results to {output_json_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    process_queries_batch(
        input_json_path=os.path.join(BASE_DIR, "Data_files", "ground_truth.json"),
        output_json_path=os.path.join(BASE_DIR, "Data_files", "prediction1.json")
    )