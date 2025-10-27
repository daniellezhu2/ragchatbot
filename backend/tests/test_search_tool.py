"""
Unit tests for CourseSearchTool

Tests the execute method and result formatting of the search tool.
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_with_query_only(self, mock_vector_store, sample_search_results):
        """Test execute with only a query parameter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test query")

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )

        # Verify result is formatted correctly
        assert isinstance(result, str)
        assert "[Test Course on AI - Lesson 1]" in result
        assert "introduction to AI" in result

    def test_execute_with_course_name(self, mock_vector_store):
        """Test execute with query and course_name"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="neural networks", course_name="AI Course")

        # Verify parameters passed correctly
        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="AI Course",
            lesson_number=None
        )

    def test_execute_with_lesson_number(self, mock_vector_store):
        """Test execute with query and lesson_number"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="deep learning", lesson_number=2)

        # Verify parameters passed correctly
        mock_vector_store.search.assert_called_once_with(
            query="deep learning",
            course_name=None,
            lesson_number=2
        )

    def test_execute_with_all_parameters(self, mock_vector_store):
        """Test execute with all parameters"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(
            query="neural networks",
            course_name="AI Course",
            lesson_number=3
        )

        # Verify all parameters passed
        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="AI Course",
            lesson_number=3
        )

    def test_execute_with_empty_results(self, mock_empty_vector_store):
        """Test execute when no results are found"""
        tool = CourseSearchTool(mock_empty_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result
        assert isinstance(result, str)

    def test_execute_with_empty_results_and_filters(self, mock_empty_vector_store):
        """Test execute with filters when no results found"""
        tool = CourseSearchTool(mock_empty_vector_store)

        result = tool.execute(
            query="test",
            course_name="Nonexistent Course",
            lesson_number=99
        )

        assert "No relevant content found" in result
        assert "Nonexistent Course" in result
        assert "lesson 99" in result

    def test_execute_with_error(self, mock_error_vector_store):
        """Test execute when vector store returns an error"""
        tool = CourseSearchTool(mock_error_vector_store)

        result = tool.execute(query="test query")

        # Should return the error message
        assert "Database connection failed" in result

    def test_format_results(self, mock_vector_store, sample_search_results):
        """Test that results are formatted with proper context headers"""
        tool = CourseSearchTool(mock_vector_store)

        # Execute to trigger formatting
        result = tool.execute(query="test")

        # Check for proper formatting
        assert "[Test Course on AI - Lesson 1]" in result
        assert "[Test Course on AI - Lesson 2]" in result
        assert "introduction to AI" in result
        assert "Deep learning" in result

    def test_source_tracking(self, mock_vector_store, sample_search_results):
        """Test that sources are tracked correctly"""
        tool = CourseSearchTool(mock_vector_store)

        # Initially no sources
        assert tool.last_sources == []

        # Execute search
        tool.execute(query="test")

        # Sources should be populated
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Test Course on AI - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

    def test_source_links_course_level(self, mock_vector_store):
        """Test source links fall back to course level when no lesson link"""
        mock_vector_store.get_lesson_link.return_value = None
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        # Should have course link as fallback
        assert tool.last_sources[0]["url"] == "https://example.com/course"


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store):
        """Test executing a tool by name"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once()

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) > 0
        assert isinstance(sources[0], dict)
        assert "text" in sources[0]

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search
        manager.execute_tool("search_course_content", query="test")
        assert len(tool.last_sources) > 0

        # Reset sources
        manager.reset_sources()

        assert tool.last_sources == []
