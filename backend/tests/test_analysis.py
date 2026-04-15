import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

@pytest.fixture
def sample_csv_file():
    csv_content = "Date,Open,High,Low,Close,Volume\n" \
                  "2023-01-01,100,105,99,102,1000\n" \
                  "2023-01-02,102,108,101,107,1200\n" \
                  "2023-01-03,107,110,105,109,1500\n" \
                  "2023-01-04,109,112,108,111,1300\n" \
                  "2023-01-05,111,115,110,114,1600\n"
    return ("test.csv", csv_content.encode(), "text/csv")

def test_analyze_swing_trading(sample_csv_file):
    response = client.post(
        "/analyze",
        files={"file": sample_csv_file},
        data={"strategy": "swing_trading", "data_type": "swing"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "swing_trading"
    assert "buy_signals" in data
    assert "sell_signals" in data

def test_analyze_unsupported_file_type():
    response = client.post(
        "/analyze",
        files={"file": ("test.txt", b"some content", "text/plain")},
        data={"strategy": "swing_trading", "data_type": "swing"}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_analyze_invalid_strategy():
    csv_content = "Date,Open,High,Low,Close,Volume\n" \
                  "2023-01-01,100,105,99,102,1000\n"
    response = client.post(
        "/analyze",
        files={"file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"strategy": "invalid_strategy", "data_type": "swing"}
    )
    assert response.status_code == 400
    assert "Invalid strategy" in response.json()["detail"]

def test_analyze_missing_columns():
    csv_content = "Date,Open,High\n" \
                  "2023-01-01,100,105\n"
    response = client.post(
        "/analyze",
        files={"file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"strategy": "swing_trading", "data_type": "swing"}
    )
    assert response.status_code == 400
    assert "Input data must contain the following columns" in response.json()["detail"]

def test_analyze_with_nan_values():
    csv_content = "Date,Open,High,Low,Close,Volume\n" \
                  "2023-01-01,100,105,99,102,1000\n" \
                  "2023-01-02,,108,101,107,1200\n" \
                  "2023-01-03,107,110,105,109,1500\n"
    response = client.post(
        "/analyze",
        files={"file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"strategy": "swing_trading", "data_type": "swing"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["buy_signals"]) >= 0  # Just checking if it runs without error
    assert len(data["sell_signals"]) >= 0
