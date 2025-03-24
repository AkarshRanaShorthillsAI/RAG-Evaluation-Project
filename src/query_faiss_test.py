import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from query_faiss import search_jobs, log_interaction

@pytest.fixture
def mock_faiss():
    """Mock FAISS search to return predefined indices."""
    with patch("query_faiss.index.search") as mock_search:
        mock_search.return_value = (np.array([[0, 1]]), np.array([[0, 1]]))
        yield mock_search

@pytest.fixture
def mock_embedding():
    """Mock the embedding function to return a dummy vector."""
    with patch("query_faiss.get_embedding", return_value=np.random.rand(1, 384).astype("float32")) as mock_embed:
        yield mock_embed

@pytest.fixture
def mock_gemini():
    """Mock Google Gemini API response."""
    with patch("query_faiss.genai.GenerativeModel") as mock_model:
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value.text = """
        **TechCorp** is Hiring for **Software Engineer** role
        They are looking for candidates with **Python, Machine Learning** and require 
        **3 years** of experience.  
        The job is based in **Remote**. 
        If you're interested, click below to apply: https://techcorp.com/job/software-engineer
        """
        mock_model.return_value = mock_instance
        yield mock_model

def test_search_jobs(mock_faiss, mock_embedding, mock_gemini):
    """Test job search functionality."""
    query = "Python developer role"
    result = search_jobs(query, k=2)

    assert "**TechCorp** is Hiring for **Software Engineer** role" in result
    assert "https://techcorp.com/job/software-engineer" in result

def test_log_interaction(tmp_path):
    """Test logging interactions."""
    log_file = tmp_path / "query_logs.json"
    
    with patch("query_faiss.LOG_FILE", str(log_file)):
        log_interaction("Test Query", [], "Sample Response")

        with open(log_file, "r") as f:
            logs = json.load(f)

    assert len(logs) == 1
    assert logs[0]["query"] == "Test Query"
    assert logs[0]["ai_response"] == "Sample Response"
