"""
Test fixtures and utilities for RAG system tests
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys
import os

# Add backend to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_course():
    """Sample course with lessons for testing"""
    return Course(
        title="Test Course on AI",
        course_link="https://example.com/ai-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Deep Learning", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Neural Networks", lesson_link="https://example.com/lesson3"),
        ]
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is an introduction to AI and machine learning concepts.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Deep learning uses neural networks with multiple layers.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are inspired by biological neurons.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample SearchResults with valid data"""
    return SearchResults(
        documents=[
            "This is an introduction to AI and machine learning concepts.",
            "Deep learning uses neural networks with multiple layers."
        ],
        metadata=[
            {"course_title": "Test Course on AI", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course on AI", "lesson_number": 2, "chunk_index": 1}
        ],
        distances=[0.2, 0.3]
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for no matches"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """SearchResults with an error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mock VectorStore that returns predictable results"""
    mock = Mock()
    mock.search.return_value = sample_search_results
    mock.get_course_link.return_value = "https://example.com/ai-course"
    mock.get_lesson_link.return_value = "https://example.com/lesson1"
    return mock


@pytest.fixture
def mock_empty_vector_store(empty_search_results):
    """Mock VectorStore that returns no results"""
    mock = Mock()
    mock.search.return_value = empty_search_results
    mock.get_course_link.return_value = None
    mock.get_lesson_link.return_value = None
    return mock


@pytest.fixture
def mock_error_vector_store(error_search_results):
    """Mock VectorStore that returns an error"""
    mock = Mock()
    mock.search.return_value = error_search_results
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generator without API calls"""
    mock = Mock()

    # Mock a simple text response (no tool use)
    simple_response = Mock()
    simple_response.content = [Mock(text="This is a test response")]
    simple_response.stop_reason = "end_turn"

    # Mock a tool use response
    tool_use_response = Mock()
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "test_tool_id_123"
    tool_block.input = {"query": "test query"}
    tool_use_response.content = [tool_block]
    tool_use_response.stop_reason = "tool_use"

    # Mock final response after tool use
    final_response = Mock()
    final_response.content = [Mock(text="Answer based on search results")]
    final_response.stop_reason = "end_turn"

    # Configure the mock to return different responses
    mock.messages.create.side_effect = [tool_use_response, final_response]

    return mock


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing"""
    mock = Mock()
    mock.execute_tool.return_value = "[Test Course] Sample search result"
    mock.get_last_sources.return_value = [
        {"text": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}
    ]
    mock.reset_sources.return_value = None
    mock.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]
    return mock


@pytest.fixture
def mock_config():
    """Mock config object for testing"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test_api_key_123"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config
