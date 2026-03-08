import pytest
from pk_ai_tools.rag_pipeline import RAGPipeline

pytestmark = pytest.mark.integration


def test_pipeline_can_start_with_ollama(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "test.txt").write_text("Paris is the capital of France.", encoding="utf-8")

    memory_dir = tmp_path / "memory"
    chroma_dir = tmp_path / "chroma"

    pipeline = RAGPipeline(
        doc_folder=str(docs),
        model_name="llama3",
        embedding_model="nomic-embed-text",
        chroma_path=str(chroma_dir),
        memory_dir=str(memory_dir),
        uuid="1",
        language="en",
    )

    assert pipeline is not None
    assert pipeline.vector_db is not None