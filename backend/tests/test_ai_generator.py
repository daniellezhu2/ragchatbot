"""
Unit tests for AIGenerator

Tests AI response generation, tool calling, and conversation history handling.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""

    def test_initialization(self):
        """Test AIGenerator initializes correctly"""
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")

        assert generator.model == "claude-3-5-sonnet-20250514"
        assert generator.base_params["model"] == "claude-3-5-sonnet-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_structure(self):
        """Test that system prompt contains key instructions"""
        system_prompt = AIGenerator.SYSTEM_PROMPT

        # Check for key phrases
        assert "search tool" in system_prompt.lower()
        assert "course content" in system_prompt.lower()
        assert "one search per query maximum" in system_prompt.lower()
        assert "general knowledge" in system_prompt.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test generating a simple response without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        # Create generator and generate response
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")
        result = generator.generate_response(query="What is machine learning?")

        # Verify response
        assert result == "This is a test response"
        mock_client.messages.create.assert_called_once()

        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["messages"][0]["content"] == "What is machine learning?"
        assert "system" in call_args.kwargs

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test that conversation history is included in system prompt"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        # Create generator
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")

        # Generate with history
        history = "User: Previous question\nAssistant: Previous answer"
        result = generator.generate_response(
            query="Follow-up question",
            conversation_history=history
        )

        # Verify history was included
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous question" in system_content
        assert "Previous answer" in system_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic_class):
        """Test that tools are passed to API"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        # Create generator
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")

        # Tools definition
        tools = [{
            "name": "search_course_content",
            "description": "Search courses",
            "input_schema": {"type": "object", "properties": {}}
        }]

        result = generator.generate_response(query="Test", tools=tools)

        # Verify tools were passed
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"]["type"] == "auto"

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic_class, mock_tool_manager):
        """Test that tool use triggers execution and follow-up call"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "test_tool_123"
        tool_block.input = {"query": "neural networks"}
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        # Second response: final answer
        final_response = Mock()
        final_response.content = [Mock(text="Answer based on search results")]
        final_response.stop_reason = "end_turn"

        # Configure mock to return both responses
        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Create generator and tools
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")
        tools = [{"name": "search_course_content"}]

        # Generate response
        result = generator.generate_response(
            query="What are neural networks?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="neural networks"
        )

        # Verify final response
        assert result == "Answer based on search results"

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_parameters_passed_correctly(self, mock_anthropic_class, mock_tool_manager):
        """Test that all tool parameters are passed correctly"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response with multiple parameters
        tool_use_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "test_tool_456"
        tool_block.input = {
            "query": "deep learning",
            "course_name": "AI Fundamentals",
            "lesson_number": 3
        }
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Generate response
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")
        result = generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify all parameters were passed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="deep learning",
            course_name="AI Fundamentals",
            lesson_number=3
        )

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_message_flow(self, mock_anthropic_class, mock_tool_manager):
        """Test that messages are properly constructed during tool execution"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_789"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Final")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Generate
        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")
        generator.generate_response(
            query="Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Check second API call (follow-up after tool use)
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Should have: user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Tool result should be properly formatted
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_789"

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_tool_manager_with_tool_use(self, mock_anthropic_class):
        """Test that tool use without tool_manager returns properly"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response but no tool manager provided
        tool_use_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        mock_client.messages.create.return_value = tool_use_response

        generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20250514")

        # This should not crash, but won't execute tools
        # The behavior depends on implementation - it should handle gracefully
        result = generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=None
        )

        # Should not raise an exception
        assert mock_client.messages.create.call_count == 1
