---
name: mcp-patterns
description: MCP (Model Context Protocol) server patterns — FastMCP server design, tool/resource/prompt definitions, authentication, testing, and deployment.
origin: local
---

# MCP Server Patterns

Design and build production MCP servers that expose tools, resources, and prompts to AI assistants.

## When to Activate

- Building an MCP server with FastMCP or the official MCP SDK
- Designing tool, resource, or prompt definitions
- Adding authentication to an MCP server
- Testing or deploying MCP servers
- Integrating an MCP server into a FastAPI app

## MCP Primitives

| Primitive | Purpose | Client sees |
|-----------|---------|-------------|
| **Tool** | Executable action the LLM can call (read DB, call API, run code) | Function with input schema |
| **Resource** | Read-only data the LLM can access (file, DB row, API response) | URI-addressed content |
| **Prompt** | Reusable prompt template with arguments | Rendered message list |

## FastMCP — Recommended Approach

```bash
pip install fastmcp
```

### Minimal Server

```python
# server.py
from fastmcp import FastMCP

mcp = FastMCP(
    name="my-server",
    version="1.0.0",
    instructions="Tools for querying the product catalog.",
)
```

## Tool Definitions

Tools are the most commonly used primitive. The LLM decides when to call them.

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("product-server")


# Simple tool — docstring becomes the description
@mcp.tool()
async def search_products(
    query: str = Field(..., description="Search term"),
    max_results: int = Field(10, ge=1, le=100, description="Max results to return"),
) -> list[dict]:
    """Search the product catalog by keyword. Returns matching products with name, price, and stock."""
    results = await db.execute(
        "SELECT id, name, price, stock FROM products WHERE name ILIKE $1 LIMIT $2",
        f"%{query}%", max_results,
    )
    return [dict(r) for r in results]


@mcp.tool()
async def get_order_status(
    order_id: str = Field(..., description="Order ID (e.g. ORD-12345)"),
) -> dict:
    """Get the current status and tracking info for an order."""
    order = await orders_service.get(order_id)
    if not order:
        return {"error": f"Order {order_id} not found"}
    return {"id": order.id, "status": order.status, "estimated_delivery": order.eta}


@mcp.tool()
async def create_refund(
    order_id: str,
    reason: str = Field(..., description="Reason for refund"),
    amount: float | None = Field(None, description="Partial amount; omit for full refund"),
) -> dict:
    """Initiate a refund for an order. Requires order to be in delivered status."""
    # Always validate before mutating state
    order = await orders_service.get(order_id)
    if not order:
        raise ValueError(f"Order {order_id} not found")
    if order.status != "delivered":
        raise ValueError(f"Can only refund delivered orders, got: {order.status}")
    return await refunds_service.create(order_id, reason, amount)
```

### Tool Design Principles

- **Descriptive docstring**: the LLM reads it to decide when and how to call the tool
- **Field descriptions**: explain each parameter clearly — be specific (units, format, valid values)
- **Return structured data**: dicts/lists, not raw strings
- **Return errors in data, not exceptions**: `{"error": "..."}` for expected failures; raise for unexpected
- **Idempotent reads**: GET-like tools can be called multiple times safely
- **Confirm before mutating**: destructive tools should have explicit confirmation logic or require a `confirm: bool` param
- **Small surface area**: one tool per action; avoid multipurpose tools with `action` flags

## Resource Definitions

Resources expose read-only data as URI-addressable content.

```python
from fastmcp import FastMCP
from fastmcp.resources import Resource

mcp = FastMCP("docs-server")


# Static resource
@mcp.resource("docs://readme")
async def get_readme() -> str:
    """Main project README."""
    return Path("README.md").read_text()


# Dynamic resource with URI template
@mcp.resource("users://{user_id}/profile")
async def get_user_profile(user_id: int) -> dict:
    """User profile data. URI: users://<id>/profile"""
    user = await users_service.get(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    return {"id": user.id, "email": user.email, "name": user.name}


@mcp.resource("reports://{date}/summary")
async def get_daily_report(date: str) -> str:
    """Daily summary report. Date format: YYYY-MM-DD"""
    report = await reports_service.get_by_date(date)
    return report.markdown_content
```

### Resource URI Conventions

- Use `scheme://path` format: `docs://`, `db://`, `files://`
- Use descriptive schemes that reflect the data domain
- Template params in `{braces}` map to function arguments
- Return `str` for text, `bytes` for binary, `dict` for JSON

## Prompt Templates

Prompts are reusable message templates with typed arguments.

```python
from fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

mcp = FastMCP("prompts-server")


@mcp.prompt()
def code_review_prompt(
    language: str,
    code: str,
    focus: str = "general quality, security, and performance",
) -> list[PromptMessage]:
    """Generate a code review prompt for a given language and code snippet."""
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""Review the following {language} code.
Focus on: {focus}

```{language}
{code}
```

Provide specific, actionable feedback.""",
            ),
        )
    ]


@mcp.prompt()
def summarize_document(
    document: str,
    format: str = "bullet points",
    max_length: str = "200 words",
) -> list[PromptMessage]:
    """Summarize a document in the specified format and length."""
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Summarize the following document in {format}, max {max_length}:\n\n{document}",
            ),
        )
    ]
```

## Authentication

```python
# Option 1: API key via header (simple, for internal tools)
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    token=settings.mcp_api_key.get_secret_value(),
)
mcp = FastMCP("secure-server", auth=auth)


# Option 2: Custom auth (per-request validation)
from fastmcp.server.auth import AuthProvider
from starlette.requests import Request

class CustomAuthProvider(AuthProvider):
    async def authenticate(self, request: Request) -> bool:
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        return await validate_token(token)   # your logic

mcp = FastMCP("secure-server", auth=CustomAuthProvider())
```

## Dependency Injection

```python
from contextlib import asynccontextmanager
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

# Inject DB session into tools via lifespan
@asynccontextmanager
async def lifespan(server: FastMCP):
    async with AsyncSessionLocal() as session:
        server.state.db = session
        yield
        await session.close()


mcp = FastMCP("db-server", lifespan=lifespan)


@mcp.tool()
async def get_user(
    user_id: int,
    ctx: Context,  # FastMCP injects this automatically
) -> dict:
    """Get a user by ID."""
    db: AsyncSession = ctx.server.state.db
    user = await db.get(User, user_id)
    return {"id": user.id, "email": user.email} if user else {"error": "Not found"}
```

## Embedding MCP in a FastAPI App

```python
# app/main.py
from fastapi import FastAPI
from fastmcp import FastMCP
from app.mcp_server import mcp

app = FastAPI()

# Mount MCP server as a sub-application
app.mount("/mcp", mcp.get_asgi_app())

# Regular FastAPI routes
@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Project Structure

```
myproject/
├── app/
│   ├── main.py
│   └── ...
├── mcp/
│   ├── __init__.py
│   ├── server.py          # FastMCP instance + lifespan
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── products.py    # product-related tools
│   │   ├── orders.py      # order-related tools
│   │   └── users.py
│   ├── resources/
│   │   ├── __init__.py
│   │   └── docs.py
│   └── prompts/
│       ├── __init__.py
│       └── templates.py
└── tests/
    └── test_mcp/
        ├── test_tools.py
        └── test_resources.py
```

## Testing MCP Servers

```python
# tests/test_mcp/test_tools.py
import pytest
from fastmcp import FastMCP, Client


@pytest.fixture
def mcp_server() -> FastMCP:
    from mcp.server import mcp  # your server instance
    return mcp


@pytest.fixture
async def client(mcp_server: FastMCP) -> Client:
    async with Client(mcp_server) as c:
        yield c


async def test_search_products_returns_results(client: Client, db):
    result = await client.call_tool("search_products", {"query": "laptop"})
    assert isinstance(result, list)
    assert len(result) > 0
    assert "name" in result[0]


async def test_search_products_empty_query(client: Client):
    result = await client.call_tool("search_products", {"query": ""})
    assert isinstance(result, list)   # graceful, not an error


async def test_get_order_status_not_found(client: Client):
    result = await client.call_tool("get_order_status", {"order_id": "ORD-99999"})
    assert "error" in result


async def test_list_tools(client: Client):
    tools = await client.list_tools()
    tool_names = {t.name for t in tools}
    assert "search_products" in tool_names
    assert "get_order_status" in tool_names


async def test_get_readme_resource(client: Client):
    content = await client.read_resource("docs://readme")
    assert len(content) > 0


async def test_code_review_prompt(client: Client):
    messages = await client.get_prompt(
        "code_review_prompt",
        {"language": "python", "code": "def foo(): pass"},
    )
    assert len(messages) == 1
    assert "python" in messages[0].content.text
```

## Deployment

### Standalone (stdio for Claude Desktop / CLI)

```python
# server.py
if __name__ == "__main__":
    mcp.run()  # uses stdio transport by default
```

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["run", "python", "server.py"],
      "env": {"DATABASE_URL": "postgresql://..."}
    }
  }
}
```

### HTTP/SSE (remote server)

```python
# server.py
if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8080)
```

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv sync --frozen
COPY . .
EXPOSE 8080
CMD ["uv", "run", "python", "server.py"]
```

## Common Anti-Patterns

| Anti-pattern | Problem | Fix |
|--------------|---------|-----|
| Vague tool description | LLM calls wrong tool or skips it | Write specific, usage-oriented docstrings |
| Raising exceptions for expected failures | Crashes the agent loop | Return `{"error": "..."}` for expected cases |
| Mutable state in tools | Race conditions under concurrent calls | Keep tools stateless; use injected DB sessions |
| Tools that do too much | LLM can't compose them | One responsibility per tool |
| No auth on HTTP server | Anyone can call your tools | Add BearerAuthProvider |
| Returning raw text from tools | Hard for LLM to parse | Return structured dicts |
| Skipping tests | Breakage only found in production | Use `Client` fixture for all tools |

## Quick Reference

| Task | Pattern |
|------|---------|
| Define a tool | `@mcp.tool()` async function with `Field` descriptions |
| Define a resource | `@mcp.resource("scheme://path")` |
| Define a prompt | `@mcp.prompt()` returning `list[PromptMessage]` |
| Inject DB session | `lifespan` + `ctx.server.state.db` |
| Add auth | `BearerAuthProvider(token=...)` |
| Embed in FastAPI | `app.mount("/mcp", mcp.get_asgi_app())` |
| Test tools | `async with Client(mcp_server) as client` |
| Run standalone | `mcp.run()` (stdio) or `mcp.run(transport="sse")` |
