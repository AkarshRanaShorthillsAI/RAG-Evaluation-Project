Project Overview

The AI-Powered Job Search System retrieves and processes job data with AI-based search and ranking. It enables users to search for jobs with natural language queries, returning the most relevant job listings.
🔹 Key Features

✅ Scrapes job postings from TimesJobs.com
✅ Stores job data using FAISS (vector search)
✅ Retrieves top job matches based on embeddings
✅ Refines job recommendations using Google Gemini AI
✅ Logs all searches and user interactions

# Project Structure

```
Evaluation Project/
│── src/                          # Source Code Directory
│   ├── scraper.py                 # Job scraping module
│   ├── job_details.py              # Extracts detailed job information
│   ├── store_job_data.py           # Converts jobs to embeddings & stores in FAISS
│   ├── query_faiss.py              # Retrieves jobs & refines results using AI
│   ├── streamlit_app.py            # Frontend UI for job search
│   ├── evaluate_ragas.py           # Evaluates retrieval accuracy
│   ├── utils.py                    # Helper functions (saving CSV, logging, etc.)
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation
```



# Installation Guide

## Step 1: Clone the Repository
```sh
git clone https://github.com/yourusername/job-search-ai.git
cd job-search-ai
```

## Step 2: Set Up Virtual Environment
```sh
python3 -m venv .venv
source .venv/bin/activate   # (On Windows use `.venv\Scripts\activate`)
pip install -r requirements.txt
```

## Step 3: Set Up Environment Variables
Create a `.env` file and add the following:
```sh
GEMINI_API_KEY="your_google_api_key"
```

# 🚀 Running the Project

## Step 1: Run the Job Scraper
```sh
python src/scraper.py
```
This will scrape job listings and save them in `scraped_jobs.csv`.

## Step 2: Store Data in FAISS
```sh
python src/store_job_data.py
```
This step converts job descriptions into embeddings and stores them in FAISS.

## Step 3: Start Streamlit UI
```sh
streamlit run src/streamlit_app.py
```
This will launch the interactive job search UI.
