from unittest.mock import MagicMock, patch
from pk_ai_tools.rag_pipeline import RAGPipeline


@patch("pk_ai_tools.rag_pipeline.Chroma")
@patch("pk_ai_tools.rag_pipeline.OllamaEmbeddings")
@patch("pk_ai_tools.rag_pipeline.ChatOllama")
@patch.object(RAGPipeline, "_setup_vector_db")
@patch.object(RAGPipeline, "_create_retriever")
@patch.object(RAGPipeline, "_create_chain")
def test_init_sets_variables(
    mock_create_chain,
    mock_create_retriever,
    mock_setup_vector_db,
    mock_chatollama,
    mock_embeddings,
    mock_chroma,
):
    mock_setup_vector_db.return_value = MagicMock()
    mock_create_retriever.return_value = MagicMock()
    mock_create_chain.return_value = MagicMock()

    pipeline = RAGPipeline(
        doc_folder="./docs",
        model_name="llama3",
        embedding_model="nomic-embed-text",
        uuid="1234",
        language="sv",
        memory_dir="./memory_test",
        max_memory_length=5,
    )

    assert pipeline.doc_folder == "./docs"
    assert pipeline.model_name == "llama3"
    assert pipeline.embedding_model == "nomic-embed-text"
    assert pipeline.uuid == "1234"
    assert pipeline.language == "sv"
    assert pipeline.max_memory_length == 5

def test_ask_returns_error_when_chain_missing():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.chain = None

    result = pipeline.ask("Hello?")
    assert result == "Error: Chain not available."

def test_build_context_from_memory():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.load_memory = MagicMock(return_value=[
        {"prompt": "Hi", "answer": "Hello"},
        {"prompt": "How are you?", "answer": "Fine"},
    ])

    result = pipeline.build_context_from_memory("123")

    assert "User: Hi" in result
    assert "AI: Hello" in result
    assert "User: How are you?" in result
    assert "AI: Fine" in result

def test_ask_invokes_chain_and_saves_memory():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.chain = MagicMock()
    pipeline.chain.invoke.return_value = "Test answer"
    pipeline.uuid = "123"
    pipeline.language = "en"
    pipeline.system_prompt = "You are a multilingual AI assistant "
    pipeline.build_context_from_memory = MagicMock(return_value="User: Hi\nAI: Hello")
    pipeline.save_memory = MagicMock()

    result = pipeline.ask("What is this?")

    assert result == "Test answer"
    pipeline.chain.invoke.assert_called_once()
    pipeline.save_memory.assert_called_once_with("123", "What is this?", "Test answer")

