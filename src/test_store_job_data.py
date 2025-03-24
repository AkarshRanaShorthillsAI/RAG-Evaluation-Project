import unittest
import pandas as pd
import numpy as np
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from store_job_data import get_embedding  # Import the function from your script

class TestJobDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the embedding model once for all tests."""
        cls.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModel.from_pretrained(cls.model_name)

    def setUp(self):
        """Create a sample dataframe to mock job listings."""
        self.sample_data = pd.DataFrame([
            {
                "Job Title": "Software Engineer",
                "Company Name": "Tech Corp",
                "Required Skills": "Python, Django",
                "Experience": 3,
                "Salary": "₹10,00,000",
                "Location": "Remote",
                "More Info": "https://example.com/job1"
            },
            {
                "Job Title": "Data Scientist",
                "Company Name": "AI Labs",
                "Required Skills": "Machine Learning, Python",
                "Experience": 5,
                "Salary": "₹15,00,000",
                "Location": "Bangalore",
                "More Info": "https://example.com/job2"
            }
        ])
    
    def test_csv_columns_exist(self):
        """Ensure the required columns exist in the CSV."""
        required_columns = {"Job Title", "Company Name", "Required Skills", "Experience", "Salary", "Location", "More Info"}
        self.assertTrue(required_columns.issubset(self.sample_data.columns), "Missing required columns in CSV.")

    def test_embedding_generation(self):
        """Check if the embedding function returns a valid NumPy array."""
        text = "Software Engineer at Tech Corp, skilled in Python and Django."
        embedding = get_embedding(text)
        
        self.assertIsInstance(embedding, np.ndarray, "Embedding should be a NumPy array.")
        self.assertEqual(embedding.shape[1], 384, "Incorrect embedding size (Expected 384 for MiniLM-L6-v2).")

    def test_job_json_structure(self):
        """Ensure job data is stored in the correct JSON format."""
        job_object = {
            "jobTitle": self.sample_data.iloc[0]["Job Title"],
            "company": self.sample_data.iloc[0]["Company Name"],
            "requiredSkills": self.sample_data.iloc[0]["Required Skills"],
            "experience": str(self.sample_data.iloc[0]["Experience"]),
            "salary": str(self.sample_data.iloc[0]["Salary"]),
            "location": self.sample_data.iloc[0]["Location"],
            "moreInfo": self.sample_data.iloc[0]["More Info"]
        }

        json_data = json.dumps([job_object])
        loaded_data = json.loads(json_data)

        self.assertEqual(len(loaded_data), 1, "JSON should contain exactly one job entry.")
        self.assertEqual(loaded_data[0]["jobTitle"], "Software Engineer", "Job Title mismatch in JSON.")

    def test_faiss_index_saving(self):
        """Check if FAISS index can be created and saved properly."""
        embeddings = np.random.rand(2, 384).astype("float32")  # Mock embeddings
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, "test_index.faiss")

        # Reload and verify
        loaded_index = faiss.read_index("test_index.faiss")
        self.assertEqual(loaded_index.ntotal, 2, "FAISS index should contain 2 embeddings.")

if __name__ == "__main__":
    unittest.main()
