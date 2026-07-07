from app.agents.router_agent import RouterAgent
from app.agents.verifier_agent import VerifierAgent


def test_router_simple_query():
    router = RouterAgent()
    result = router.run(question="What year was the company founded?")
    assert result["route"] == "simple"


def test_router_complex_query():
    router = RouterAgent()
    result = router.run(
        question="Compare the revenue growth versus the competitors and explain why it changed"
    )
    assert result["route"] == "complex"


def test_verifier_flags_unsupported_claim():
    verifier = VerifierAgent()
    chunks = [{"chunk_id": "doc1_chunk0", "text": "Aurora Robotics was founded in 2019."}]
    answer = "Aurora Robotics was founded in 2019. The company was later acquired by Google for ten billion dollars."
    result = verifier.run(answer=answer, chunks=chunks)
    assert result["verification_status"] == "unverified_claims_found"
    assert len(result["unsupported_claims"]) >= 1


def test_verifier_accepts_supported_claim():
    verifier = VerifierAgent()
    chunks = [{"chunk_id": "doc1_chunk0", "text": "Aurora Robotics was founded in 2019 and builds warehouse drones."}]
    answer = "Aurora Robotics was founded in 2019 and builds warehouse drones."
    result = verifier.run(answer=answer, chunks=chunks)
    assert result["verification_status"] == "verified"
    assert result["unsupported_claims"] == []
