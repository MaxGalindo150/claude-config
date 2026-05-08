---
name: langfuse-observability
description: LLM observability with Langfuse — tracing LLM calls, spans, prompt versioning, cost tracking, evaluation datasets, and multi-tenant trace isolation in FastAPI/Python applications.
origin: local
---

# Langfuse Observability Patterns

Production patterns for tracing, evaluating, and monitoring LLM pipelines with Langfuse.

## When to Activate

- Adding observability to LLM calls (Anthropic, OpenAI)
- Tracing multi-step agent pipelines
- Managing and versioning prompts
- Tracking cost and latency per tenant/user
- Building evaluation datasets from production traces
- Debugging LLM failures or unexpected outputs

## Setup

```python
# app/core/observability.py
from langfuse import Langfuse
from langfuse.decorators import langfuse_context
from functools import lru_cache

@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,  # Default: cloud.langfuse.com
    )
```

```bash
# .env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted
```

## Tracing with Decorators (Recommended)

### Basic Trace

```python
from langfuse.decorators import observe, langfuse_context

@observe()
async def generate_reply(conversation_id: str, messages: list[dict]) -> str:
    # Automatically creates a trace named "generate_reply"
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text
```

### Nested Spans

```python
@observe()  # Creates root trace
async def handle_user_message(conversation_id: str, message: str):
    # Each @observe() call within creates a child span
    context = await retrieve_context(conversation_id, message)
    reply = await generate_reply(conversation_id, context)
    await save_reply(conversation_id, reply)
    return reply

@observe(name="retrieve_context")  # Custom span name
async def retrieve_context(conversation_id: str, query: str) -> list[dict]:
    # Embed + vector search
    ...
```

### Add Metadata to Trace

```python
@observe()
async def generate_reply(conversation_id: str, messages: list[dict]) -> str:
    # Tag the current trace with metadata
    langfuse_context.update_current_trace(
        name="generate_reply",
        user_id=current_user.uid,
        session_id=conversation_id,
        tags=["production", "v2"],
        metadata={
            "tenant_id": current_tenant.id,
            "model": "claude-sonnet-4-6",
            "message_count": len(messages),
        },
    )

    response = await anthropic_client.messages.create(...)
    return response.content[0].text
```

## Native SDK Integration

### Anthropic with Langfuse

```python
from langfuse.anthropic import anthropic as langfuse_anthropic

# Drop-in replacement — wraps the Anthropic client
client = langfuse_anthropic.Anthropic()

# All calls are automatically traced
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
    metadata={
        "langfuse_user_id": user_id,
        "langfuse_session_id": session_id,
        "langfuse_tags": ["production"],
    },
)
```

### OpenAI with Langfuse

```python
from langfuse.openai import openai as langfuse_openai

client = langfuse_openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    metadata={
        "langfuse_user_id": user_id,
        "langfuse_session_id": session_id,
    },
)
```

## Multi-Tenant Trace Isolation

Always tag traces with `user_id` and a custom `tenant_id` metadata field. This enables per-tenant cost analysis.

```python
@observe()
async def process_request(
    request_data: dict,
    user: FirebaseUser,
    tenant: Tenant,
):
    langfuse_context.update_current_trace(
        user_id=user.uid,                     # Langfuse user_id (built-in)
        session_id=request_data.get("conversation_id"),
        metadata={
            "tenant_id": tenant.id,           # Custom — for filtering in UI
            "tenant_plan": tenant.plan,
            "environment": settings.ENVIRONMENT,
        },
        tags=[tenant.plan, settings.ENVIRONMENT],
    )
    ...
```

## Prompt Management

### Fetch Prompt from Langfuse

```python
from langfuse import Langfuse

langfuse = get_langfuse()

def get_prompt(name: str, variables: dict) -> str:
    """Fetch versioned prompt from Langfuse. Falls back to latest if not pinned."""
    prompt = langfuse.get_prompt(name)  # Fetches with caching (5min TTL)
    return prompt.compile(**variables)

# Usage
system_prompt = get_prompt("support_agent_v2", {
    "tenant_name": tenant.name,
    "language": "es",
})
```

### Register Prompt via API

```python
langfuse.create_prompt(
    name="support_agent_v2",
    prompt="You are a support agent for {{tenant_name}}. Reply in {{language}}.",
    is_active=True,
    labels=["production"],
)
```

## Cost Tracking

Langfuse automatically tracks token usage and cost when using the native integrations. For custom models or manual tracking:

```python
@observe()
async def call_model(prompt: str, model: str) -> str:
    start = time.time()
    response = await my_custom_llm(prompt)
    latency_ms = int((time.time() - start) * 1000)

    # Manually log usage
    langfuse_context.update_current_observation(
        usage={
            "input": count_tokens(prompt),
            "output": count_tokens(response),
            "unit": "TOKENS",
        },
        model=model,
        model_parameters={"temperature": 0.7, "max_tokens": 1024},
        latency=latency_ms,
    )
    return response
```

## Evaluation / Scoring

```python
from langfuse import Langfuse

langfuse = get_langfuse()

# Score a specific trace after human review or automated eval
async def score_response(trace_id: str, score: float, comment: str | None = None):
    langfuse.score(
        trace_id=trace_id,
        name="quality",           # Metric name
        value=score,              # 0.0 to 1.0
        comment=comment,
        data_type="NUMERIC",
    )

# Boolean scoring (correct / incorrect)
async def score_correctness(trace_id: str, correct: bool):
    langfuse.score(
        trace_id=trace_id,
        name="correct",
        value=1 if correct else 0,
        data_type="BOOLEAN",
    )
```

### Automated Evaluation with LLM-as-Judge

```python
@observe(name="eval_response_quality")
async def evaluate_quality(question: str, answer: str, trace_id: str) -> float:
    eval_prompt = f"""Rate the quality of this answer from 0-10.

Question: {question}
Answer: {answer}

Reply with only a number."""

    response = await anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",  # Cheap model for eval
        max_tokens=10,
        messages=[{"role": "user", "content": eval_prompt}],
    )
    score = float(response.content[0].text.strip()) / 10

    langfuse.score(trace_id=trace_id, name="llm_quality", value=score)
    return score
```

## FastAPI Middleware — Request-Level Tracing

```python
from langfuse.decorators import langfuse_context
import uuid

@app.middleware("http")
async def langfuse_trace_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.langfuse_trace_id = trace_id

    # Set trace metadata available to all @observe() calls in this request
    langfuse_context.update_current_trace(
        metadata={
            "path": request.url.path,
            "method": request.method,
        },
    )

    response = await call_next(request)

    # Flush at end of request (important for serverless)
    langfuse_context.flush()

    return response
```

## Flush in Serverless Environments

Cloud Run, Lambda, and similar serverless environments may kill the process before background flushes complete.

```python
# app/main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Flush all pending Langfuse events before shutdown
    get_langfuse().flush()

# Per-request flush for short-lived invocations
@router.post("/webhooks/process")
async def process_webhook(payload: dict):
    await handle_payload(payload)
    get_langfuse().flush()  # Ensure traces are sent before response
    return {"status": "ok"}
```

## Useful Queries in Langfuse UI

- **Cost by tenant:** Filter by `metadata.tenant_id`, group by date
- **Latency P95 by model:** Group traces by `model`, chart `latency`
- **Error rate:** Filter by `level=ERROR`, count by day
- **Prompt comparison:** Create experiments with different prompt versions

## Debug Mode

```python
import logging
logging.getLogger("langfuse").setLevel(logging.DEBUG)  # Verbose SDK logs

# Or check if traces are being sent
langfuse = get_langfuse()
print(langfuse.auth_check())  # Should print {"status": "success"}
```

## Reference Skills

- LLM call patterns → skill: `llm-engineering`
- FastAPI middleware → skill: `fastapi-patterns`
- Multi-tenant context → skill: `multi-tenant-saas`
- Cost-aware pipelines → skill: `cost-aware-llm-pipeline`
