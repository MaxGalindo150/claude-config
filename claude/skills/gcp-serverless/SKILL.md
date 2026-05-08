---
name: gcp-serverless
description: GCP serverless deployment patterns — Cloud Run, Cloud Tasks, Secret Manager, Artifact Registry, Cloud Storage, IAM, and cost-aware architecture for production Python services.
origin: local
---

# GCP Serverless Patterns

Production patterns for deploying Python/containerized services on Google Cloud Platform's serverless stack.

## When to Activate

- Deploying services to Cloud Run
- Setting up async background jobs with Cloud Tasks
- Managing secrets with Secret Manager
- Building CI/CD pipelines with Artifact Registry
- Configuring IAM service accounts
- Debugging Cloud Run cold starts, timeouts, or memory issues

## Cloud Run

### Dockerfile Best Practices

```dockerfile
# Multi-stage build — keep final image small
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

FROM python:3.11-slim
WORKDIR /app
# Never run as root
RUN adduser --disabled-password --gecos "" appuser
COPY --from=builder /app/.venv /app/.venv
COPY . .
USER appuser
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "cheo.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Cloud Run Service Configuration

```yaml
# Key flags for gcloud run deploy
--memory 2Gi
--cpu 1
--timeout 300
--concurrency 80
--min-instances 0        # Scale to zero in staging
--min-instances 1        # Keep warm in production
--max-instances 10
--service-account sa-name@project.iam.gserviceaccount.com
--set-env-vars ENV=production
--set-secrets DB_URL=db-url:latest,SECRET_KEY=secret-key:latest
```

### Startup & Health Checks

```python
# FastAPI lifespan for graceful startup/shutdown
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm connection pool before receiving traffic
    await db.connect()
    yield
    # Shutdown: drain connections
    await db.disconnect()

app = FastAPI(lifespan=lifespan)

# Health endpoint — must respond within 10s or Cloud Run won't route traffic
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Cold Start Optimization

- Keep image size under 500MB — use slim base images
- Lazy-load heavy imports (torch, transformers) inside functions
- Use `--min-instances 1` for latency-sensitive services
- Set `--cpu-boost` during startup for CPU-intensive init
- Connection pool: set `pool_size` low (2–5) — Cloud Run scales horizontally

## Cloud Tasks

### Enqueue a Task (HTTP target)

```python
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
import datetime, json

client = tasks_v2.CloudTasksAsyncClient()

async def enqueue_task(
    payload: dict,
    queue: str = "my-queue",
    delay_seconds: int = 0,
) -> str:
    parent = client.queue_path(PROJECT_ID, REGION, queue)

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{BASE_URL}/tasks/process",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode(),
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
            },
        }
    }

    if delay_seconds:
        schedule_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=delay_seconds)
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(schedule_time)
        task["schedule_time"] = timestamp

    response = await client.create_task(parent=parent, task=task)
    return response.name
```

### Task Handler (FastAPI)

```python
import hmac
from fastapi import Request, HTTPException

@router.post("/tasks/process")
async def process_task(request: Request):
    # Verify the request came from Cloud Tasks (OIDC token validated by Cloud Run)
    # Cloud Run handles OIDC verification automatically if SA has roles/run.invoker

    body = await request.json()
    task_name = request.headers.get("X-CloudTasks-TaskName")
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))

    if retry_count > 5:
        # Poison pill — don't retry forever
        logger.error(f"Task {task_name} exceeded retry limit")
        return {"status": "dropped"}

    await handle_task(body)
    return {"status": "ok"}
```

### Queue Configuration (gcloud)

```bash
gcloud tasks queues create my-queue \
  --max-attempts=5 \
  --max-retry-duration=3600s \
  --min-backoff=10s \
  --max-backoff=300s \
  --max-doublings=4 \
  --location=us-central1
```

## Secret Manager

### Access Secrets at Startup

```python
from google.cloud import secretmanager

def get_secret(name: str, project_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{name}/versions/latest"
    response = client.access_secret_version(name=secret_name)
    return response.payload.data.decode("utf-8")
```

### Preferred: Inject via --set-secrets

```bash
# Mount secret as env var — no SDK calls at runtime, version-pinned
gcloud run deploy my-service \
  --set-secrets DB_URL=database-url:latest \
  --set-secrets SECRET_KEY=secret-key:2  # pin to version 2
```

### IAM for Secret Access

```bash
gcloud secrets add-iam-policy-binding my-secret \
  --member="serviceAccount:sa@project.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Artifact Registry

### Build and Push (CI/CD)

```bash
# Authenticate
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push
IMAGE="us-central1-docker.pkg.dev/PROJECT/REPO/SERVICE:$GITHUB_SHA"
docker build -t $IMAGE .
docker push $IMAGE

# Deploy
gcloud run deploy my-service \
  --image $IMAGE \
  --region us-central1
```

### GitHub Actions Pattern

```yaml
- name: Authenticate to Google Cloud
  uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
    service_account: ${{ secrets.SA_EMAIL }}

- name: Build and push
  run: |
    IMAGE="${{ env.REGISTRY }}/${{ env.SERVICE }}:${{ github.sha }}"
    docker build -t $IMAGE .
    docker push $IMAGE
    echo "IMAGE=$IMAGE" >> $GITHUB_ENV

- name: Deploy to Cloud Run
  run: |
    gcloud run deploy ${{ env.SERVICE }} \
      --image ${{ env.IMAGE }} \
      --region ${{ env.REGION }} \
      --quiet
```

## IAM — Least Privilege

### Service Account per Service

```bash
# Create dedicated SA
gcloud iam service-accounts create sa-my-service \
  --display-name="My Service SA"

# Grant only what's needed
gcloud projects add-iam-policy-binding PROJECT \
  --member="serviceAccount:sa-my-service@PROJECT.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Never use default compute SA in production — too broad
```

### Required Roles by Service

| Service | Required Role |
|---------|--------------|
| Cloud SQL | `roles/cloudsql.client` |
| Secret Manager | `roles/secretmanager.secretAccessor` |
| Cloud Tasks (enqueue) | `roles/cloudtasks.enqueuer` |
| Cloud Tasks (invoke) | `roles/run.invoker` (on the target service) |
| Artifact Registry (push) | `roles/artifactregistry.writer` |
| Pub/Sub (publish) | `roles/pubsub.publisher` |

## Cost Control

- Enable **budget alerts** at 50%/90%/100% of monthly budget
- Use `--min-instances 0` everywhere except production critical services
- Set `--max-instances` — uncapped scaling = uncapped cost
- Cloud Tasks retries can amplify costs — always set `--max-attempts`
- Cloud Run CPU only billed while handling requests (default) — don't use `--cpu-always-allocated` unless needed

## Logging & Observability

```python
import structlog, json

# Structured logging — Cloud Logging indexes JSON automatically
logger = structlog.get_logger()

# Always include trace context for Cloud Trace correlation
@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_header = request.headers.get("X-Cloud-Trace-Context", "")
    with structlog.contextvars.bound_contextvars(
        trace=trace_header,
        path=request.url.path,
        method=request.method,
    ):
        response = await call_next(request)
        logger.info("request", status=response.status_code)
        return response
```

```bash
# Stream logs
gcloud run services logs read my-service --region us-central1 --tail=100

# Filter errors
gcloud logging read 'resource.type="cloud_run_revision" severity>=ERROR' \
  --project=PROJECT --limit=50
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Cold start timeout | Slow startup code | Move heavy init to lifespan, set `--cpu-boost` |
| 504 Gateway Timeout | Handler exceeds `--timeout` | Offload to Cloud Tasks |
| OOM killed | Memory limit | Increase `--memory` or reduce pool sizes |
| 403 on task handler | SA missing `roles/run.invoker` | Grant to Cloud Tasks SA |
| Image not found | AR permissions | Grant `roles/artifactregistry.reader` to Cloud Run SA |
