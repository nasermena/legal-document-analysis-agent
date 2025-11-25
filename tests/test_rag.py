import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.rag import ingest_text, retrieve


def test_ingest_and_retrieve_roundtrip():
    raw = (
        "SERVICE AGREEMENT\n\n"
        "3. Term and Termination\n"
        "This Agreement lasts 6 months. Either party may terminate with 30 days written notice.\n"
        "4. Confidentiality\n"
        "Each party must keep non-public information confidential.\n"
    )

    # 1) ingest
    doc_id = ingest_text(raw)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    # 2) retrieve (المهم ما يكسر ويرجع list)
    docs = retrieve(doc_id, "termination conditions", k=3)

    # فقط نتأكد إن النتيجة list من Documents (حتى لو فاضية)
    assert isinstance(docs, list)
    # optional: لو مش فاضية، يكون فيها نص من العقد
    if docs:
        joined = " ".join(d.page_content for d in docs).lower()
        assert "termination" in joined or "notice" in joined