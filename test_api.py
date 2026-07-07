import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.database import init_db
from app.main import app

init_db()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_and_query_pipeline():
    sample_text = (
        b"Aurora Robotics was founded in 2019. It builds autonomous "
        b"warehouse inventory drones and reported revenue of 4.2 million "
        b"dollars in 2022."
    )
    files = {"file": ("test_doc.txt", io.BytesIO(sample_text), "text/plain")}
    upload_response = client.post("/upload", files=files)
    assert upload_response.status_code == 200
    body = upload_response.json()
    assert body["document"]["filename"] == "test_doc.txt"
    assert body["document"]["num_chunks"] >= 1

    query_response = client.post("/query", json={"question": "When was Aurora Robotics founded?"})
    assert query_response.status_code == 200
    result = query_response.json()
    assert result["route"] in ("simple", "complex")
    assert "answer" in result
    assert isinstance(result["sources"], list)


def test_empty_query_rejected():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400
