"""
Integration tests for the RAG system

Tests end-to-end query processing with real components.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_system import RAGSystem
from vector_store import VectorStore, SearchResults
from config import Config


class TestRAGSystemIntegration:
    """Integration tests for RAG system"""

    @patch('rag_system.AIGenerator')
    def test_rag_system_initialization(self, mock_ai_generator_class, mock_config):
        """Test that RAG system initializes all components"""
        rag = RAGSystem(mock_config)

        # Check components are initialized
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None

    @patch('rag_system.AIGenerator')
    def test_query_without_session(self, mock_ai_generator_class, mock_config):
        """Test query processing without session ID"""
        # Setup mock AI generator
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "This is a test answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)

        # Execute query
        answer, sources = rag.query("What is machine learning?")

        # Verify AI generator was called
        mock_ai_instance.generate_response.assert_called_once()

        # Verify response
        assert answer == "This is a test answer"
        assert isinstance(sources, list)

    @patch('rag_system.AIGenerator')
    def test_query_with_session(self, mock_ai_generator_class, mock_config):
        """Test query processing with session ID"""
        # Setup mock
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer with context"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)

        # Create session
        session_id = rag.session_manager.create_session()

        # Execute query
        answer, sources = rag.query("What is deep learning?", session_id=session_id)

        # Verify conversation history was passed
        call_args = mock_ai_instance.generate_response.call_args
        # First query should have no history
        assert call_args.kwargs.get("conversation_history") is None or call_args.kwargs.get("conversation_history") == ""

        # Execute second query
        answer2, sources2 = rag.query("Tell me more", session_id=session_id)

        # Second query should have history
        call_args2 = mock_ai_instance.generate_response.call_args
        history = call_args2.kwargs.get("conversation_history")
        assert history is not None
        assert "What is deep learning?" in history

    @patch('rag_system.AIGenerator')
    def test_tools_are_passed_to_ai(self, mock_ai_generator_class, mock_config):
        """Test that tools are passed to AI generator"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)
        rag.query("What is neural networks?")

        # Verify tools were passed
        call_args = mock_ai_instance.generate_response.call_args
        tools = call_args.kwargs.get("tools")
        assert tools is not None
        assert len(tools) > 0
        assert tools[0]["name"] == "search_course_content"

    @patch('rag_system.AIGenerator')
    def test_tool_manager_is_passed(self, mock_ai_generator_class, mock_config):
        """Test that tool manager is passed for tool execution"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)
        rag.query("Test query")

        # Verify tool manager was passed
        call_args = mock_ai_instance.generate_response.call_args
        tool_manager = call_args.kwargs.get("tool_manager")
        assert tool_manager is not None
        assert tool_manager == rag.tool_manager

    @patch('rag_system.AIGenerator')
    def test_sources_retrieved_after_query(self, mock_ai_generator_class, mock_config):
        """Test that sources are retrieved from tool manager after query"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)

        # Mock tool manager to return sources
        mock_sources = [{"text": "Test Course", "url": "https://example.com"}]
        rag.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        rag.tool_manager.reset_sources = Mock()

        answer, sources = rag.query("Test query")

        # Verify sources were retrieved
        rag.tool_manager.get_last_sources.assert_called_once()
        assert sources == mock_sources

        # Verify sources were reset
        rag.tool_manager.reset_sources.assert_called_once()

    @patch('rag_system.AIGenerator')
    def test_query_prompt_format(self, mock_ai_generator_class, mock_config):
        """Test that query is formatted correctly for AI"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)
        user_query = "What is machine learning?"
        rag.query(user_query)

        # Check the prompt format
        call_args = mock_ai_instance.generate_response.call_args
        query_arg = call_args.kwargs.get("query")
        assert user_query in query_arg
        assert "Answer this question about course materials:" in query_arg

    @patch('rag_system.AIGenerator')
    def test_session_history_updated(self, mock_ai_generator_class, mock_config):
        """Test that session history is updated after query"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Test answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)
        session_id = rag.session_manager.create_session()

        # Execute query
        rag.query("Test question", session_id=session_id)

        # Check history was updated
        history = rag.session_manager.get_conversation_history(session_id)
        assert "Test question" in history
        assert "Test answer" in history


class TestRAGSystemWithRealVectorStore:
    """Integration tests using a real (temporary) vector store"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def real_config(self, temp_chroma_path):
        """Config with temporary ChromaDB path"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        return config

    @patch('rag_system.AIGenerator')
    def test_vector_store_search_integration(self, mock_ai_generator_class, real_config, sample_course, sample_chunks):
        """Test that vector store search works in the RAG system"""
        # Setup mock AI
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(real_config)

        # Add course data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_chunks)

        # Perform search directly on vector store
        results = rag.vector_store.search(query="neural networks")

        # Verify search returns results
        assert not results.is_empty()
        assert len(results.documents) > 0

    @patch('rag_system.AIGenerator')
    def test_search_tool_with_real_vector_store(self, mock_ai_generator_class, real_config, sample_course, sample_chunks):
        """Test search tool with real vector store data"""
        # Setup mock AI
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(real_config)

        # Add course data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_chunks)

        # Execute search tool directly
        result = rag.search_tool.execute(query="deep learning")

        # Verify result
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error message
        assert "No relevant content found" not in result or "Deep learning" in result

    @patch('rag_system.AIGenerator')
    def test_empty_database_search(self, mock_ai_generator_class, real_config):
        """Test search when database is empty"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(real_config)

        # Search empty database
        result = rag.search_tool.execute(query="test")

        # Should return "no content found" message
        assert "No relevant content found" in result


class TestRAGSystemErrorHandling:
    """Test error handling in the RAG system"""

    @patch('rag_system.AIGenerator')
    def test_ai_generator_exception(self, mock_ai_generator_class, mock_config):
        """Test handling of AI generator exceptions"""
        # Make AI generator raise an exception
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.side_effect = Exception("API Error")
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)

        # This should raise an exception (not caught internally)
        with pytest.raises(Exception) as exc_info:
            rag.query("Test query")

        assert "API Error" in str(exc_info.value)

    @patch('rag_system.AIGenerator')
    def test_vector_store_error_propagation(self, mock_ai_generator_class, mock_config):
        """Test that vector store errors are handled by search tool"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Answer"
        mock_ai_generator_class.return_value = mock_ai_instance

        rag = RAGSystem(mock_config)

        # Make vector store return error
        error_results = SearchResults.empty("Database error")
        rag.vector_store.search = Mock(return_value=error_results)

        # Execute search tool
        result = rag.search_tool.execute(query="test")

        # Should return the error message
        assert "Database error" in result
