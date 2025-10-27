# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot system for querying course materials. It implements an **agentic, tool-based RAG pattern** where Claude Sonnet 4 autonomously decides when and how to search course content.

**Technology Stack:**
- Backend: Python 3.13+ with FastAPI + Uvicorn
- Vector Database: ChromaDB with persistent storage
- Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
- AI: Anthropic Claude Sonnet 4 with tool calling
- Frontend: Vanilla JavaScript/HTML/CSS

## Development Commands

### Running the Application

```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points:
# - Web interface: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

### Dependency Management

```bash
# Install/sync dependencies
uv sync

# Add new dependency
# Edit pyproject.toml, then run: uv sync
```

### Environment Setup

Create `.env` in project root:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Architecture

### Core RAG Flow

The system implements a 3-phase RAG methodology:

1. **RETRIEVE**: Vector search in ChromaDB for relevant course chunks
2. **AUGMENT**: Claude receives search results as tool execution responses
3. **GENERATE**: Claude synthesizes answer from retrieved context

### Component Hierarchy

```
RAGSystem (rag_system.py)
├── DocumentProcessor (document_processor.py) - Parse & chunk course documents
├── VectorStore (vector_store.py) - ChromaDB wrapper with 2 collections
├── AIGenerator (ai_generator.py) - Claude API integration
├── SessionManager (session_manager.py) - Conversation history
└── ToolManager + CourseSearchTool (search_tools.py) - Tool definitions & execution
```

### Request Flow

```
User Query (frontend/script.js)
    ↓ POST /api/query
FastAPI Endpoint (app.py:56)
    ↓ rag_system.query()
RAGSystem (rag_system.py:102)
    ↓ ai_generator.generate_response()
AIGenerator (ai_generator.py:43)
    ↓ Claude API with tools
Claude decides to use search_course_content tool
    ↓ tool_manager.execute_tool()
CourseSearchTool (search_tools.py:52)
    ↓ vector_store.search()
VectorStore (vector_store.py:61)
    ├─ Resolve course name via semantic search on catalog
    ├─ Build filters (course + lesson)
    └─ Query content collection
ChromaDB returns top 5 similar chunks
    ↓ Format with course/lesson context
    ↓ Return to Claude as tool result
Claude generates final answer
    ↓ RAGSystem extracts sources
    ↓ Update session history
Response returned to frontend
```

### Two-Collection Vector Store Pattern

**Why two collections?** Separation of concerns for different search needs.

**Collection 1: `course_catalog`** (vector_store.py:51)
- **Purpose**: Course metadata for semantic name resolution
- **Documents**: Course titles (stored as text for embedding)
- **Metadata**: title, instructor, course_link, lessons_json, lesson_count
- **Use Case**: When user says "MCP course", fuzzy match to "Building Agentic RAG with Claude"

**Collection 2: `course_content`** (vector_store.py:52)
- **Purpose**: Actual course material chunks for content search
- **Documents**: Text chunks (800 chars with 100 char overlap)
- **Metadata**: course_title, lesson_number, chunk_index
- **Use Case**: Semantic search for relevant content to answer queries

**Search Flow:**
1. If `course_name` provided → semantic search `course_catalog` to resolve exact title
2. Build filter dict with resolved title and/or lesson_number
3. Search `course_content` with filters for relevant chunks

### Tool-Based (Agentic) RAG Pattern

**Key Difference from Naive RAG:** Claude decides whether to search, not forced retrieval.

**System Prompt** (ai_generator.py:8-29):
- "Use the search tool **only** for questions about specific course content"
- "**General knowledge questions**: Answer using existing knowledge without searching"
- "**One search per query maximum**"

**Tool Definition** (search_tools.py:27-50):
```python
{
    "name": "search_course_content",
    "description": "Search course materials with smart course name matching and lesson filtering",
    "input_schema": {
        "properties": {
            "query": {"type": "string"},        # Required
            "course_name": {"type": "string"},  # Optional fuzzy match
            "lesson_number": {"type": "integer"} # Optional filter
        }
    }
}
```

**Behavior Examples:**
- "What is machine learning?" → Claude answers directly (no tool use)
- "What's in lesson 3?" → Claude uses tool: `search_course_content(query="lesson 3 content")`
- "Neural networks in the MCP course" → `search_course_content(query="neural networks", course_name="MCP")`

### Document Format

Course documents in `docs/` must follow this structure:

```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor name]

Lesson 1: [lesson title]
Lesson Link: [lesson url]

[Lesson content...]

Lesson 2: [next lesson title]
...
```

**Parsing Logic** (document_processor.py:97-259):
1. Lines 1-3: Course metadata extraction
2. Remaining lines: Lesson markers (`Lesson \d+:`) split content
3. Each lesson's content is chunked with 800 char size, 100 char overlap
4. First chunk of each lesson prefixed with context: `"Lesson {n} content: {chunk}"`
5. Creates `Course` object with `Lesson` list and `CourseChunk` list

### Session Management

**Purpose:** Maintain conversation context across queries.

**Implementation** (session_manager.py):
- Sessions stored in memory: `Dict[str, List[Message]]`
- Max history: 2 exchanges (4 messages total) - controlled by `config.MAX_HISTORY`
- History formatted and injected into Claude's system prompt
- Session ID returned to frontend and passed in subsequent requests

**Why limited history?** Balance context relevance with token efficiency.

### Configuration

All tunable parameters in `backend/config.py`:

```python
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
MAX_RESULTS = 5            # Top-K search results
MAX_HISTORY = 2            # Conversation exchanges to remember
CHROMA_PATH = "./chroma_db"
```

**When to adjust:**
- `CHUNK_SIZE`: Larger for technical content, smaller for conversational
- `MAX_RESULTS`: More for comprehensive answers, fewer for focused responses
- `MAX_HISTORY`: Longer for complex multi-turn conversations

## Adding Features

### Adding a New Tool

1. Create tool class inheriting from `Tool` (search_tools.py:6)
2. Implement `get_tool_definition()` and `execute(**kwargs)`
3. Register in RAGSystem: `self.tool_manager.register_tool(new_tool)`
4. Update AI system prompt if needed

### Adding New API Endpoints

Pattern in `app.py`:
```python
@app.post("/api/endpoint", response_model=ResponseModel)
async def endpoint_name(request: RequestModel):
    result = rag_system.method()
    return ResponseModel(data=result)
```

Always use Pydantic models for request/response validation.

### Modifying Search Behavior

**Vector Store** (vector_store.py:61):
- `search()` method handles all search logic
- Modify `_build_filter()` for new filter types
- Adjust `max_results` via config or method parameter

**Search Tool** (search_tools.py:52):
- `execute()` calls vector store and formats results
- Modify `_format_results()` to change output structure
- Update tool definition schema for new parameters

### Frontend Modifications

**Key Files:**
- `frontend/index.html` - Structure and layout
- `frontend/script.js` - API calls and UI logic (sendMessage at line 45)
- `frontend/style.css` - Styling

**API Integration Pattern:**
```javascript
const response = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query, session_id})
});
const data = await response.json();
// data.answer, data.sources, data.session_id
```

## Startup Behavior

**app.py:88 - @app.on_event("startup")**

On server start:
1. Check if `../docs` directory exists
2. Call `rag_system.add_course_folder(docs_path, clear_existing=False)`
3. Process all `.txt`, `.pdf`, `.docx` files
4. Skip courses already in vector store (checks by title)
5. Add new courses to both collections
6. Print summary: "Loaded {n} courses with {m} chunks"

**Idempotent Loading:** Re-running won't duplicate courses (title-based deduplication).

## Data Persistence

**ChromaDB Storage:** `backend/chroma_db/` (gitignored)
- Persists across server restarts
- Two collection directories: `course_catalog` and `course_content`
- To reset: Delete `chroma_db/` folder or use `vector_store.clear_all_data()`

**No Database Migrations:** ChromaDB is schema-less. Metadata changes require rebuild:
```python
rag_system.add_course_folder(docs_path, clear_existing=True)
```

## Important Patterns

### Error Handling in Vector Search

`SearchResults` dataclass (vector_store.py:8-32):
```python
@dataclass
class SearchResults:
    documents: List[str]
    metadata: List[Dict]
    distances: List[float]
    error: Optional[str] = None  # Graceful error handling
```

Always check `results.error` and `results.is_empty()` before formatting.

### Tool Result Format

**Critical for Claude:** Search results must include course/lesson context in formatted output.

Pattern (search_tools.py:88-114):
```python
formatted = []
for doc, meta in zip(results.documents, results.metadata):
    header = f"[{meta['course_title']}"
    if lesson_num := meta.get('lesson_number'):
        header += f" - Lesson {lesson_num}"
    header += "]"
    formatted.append(f"{header}\n{doc}")

return "\n\n".join(formatted)
```

This structure helps Claude cite sources accurately.

### Conversation History Format

**Injected into system prompt** (ai_generator.py:61-64):
```python
system_content = (
    f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
    if conversation_history else self.SYSTEM_PROMPT
)
```

History format from SessionManager (session_manager.py:42-56):
```
User: [previous question]
Assistant: [previous answer]
User: [earlier question]
Assistant: [earlier answer]
```

## Common Gotchas

1. **Missing .env file**: Server starts but API calls fail with Anthropic auth errors
2. **Course name mismatches**: Semantic search on catalog resolves fuzzy names, but needs at least partial word match
3. **Empty search results**: Check if docs loaded on startup (`Loaded X courses` in logs)
4. **Tool not called**: Claude decides not to search - check system prompt alignment with query type
5. **Source tracking**: Sources must be retrieved via `tool_manager.get_last_sources()` after generation, then reset
6. **Session history**: Limited to MAX_HISTORY exchanges - older context is lost
7. **ChromaDB warnings**: Resource tracker warnings filtered in app.py:1-2 (harmless)

## File References for Key Logic

| Functionality | File:Line |
|---------------|-----------|
| Main API endpoint | app.py:56 |
| RAG query orchestration | rag_system.py:102 |
| Claude API call with tools | ai_generator.py:43 |
| Tool execution handler | ai_generator.py:89 |
| Search tool implementation | search_tools.py:52 |
| Vector search with filters | vector_store.py:61 |
| Course name resolution | vector_store.py:102 |
| Document parsing | document_processor.py:97 |
| Text chunking logic | document_processor.py:25 |
| Session history format | session_manager.py:42 |
| Frontend query send | frontend/script.js:45 |
| Course loading on startup | app.py:88 |
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use UV to run Python files