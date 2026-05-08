---
name: multi-tenant-saas
description: Multi-tenant SaaS patterns — tenant isolation strategies, row-level security, tenant-aware SQLAlchemy models, RBAC, subdomain routing, and Alembic migrations for shared schema.
origin: local
---

# Multi-Tenant SaaS Patterns

Patterns for building secure, scalable multi-tenant applications with proper data isolation.

## When to Activate

- Designing tenant isolation in a shared database
- Adding tenant context to SQLAlchemy models
- Implementing Row-Level Security in PostgreSQL
- Building RBAC for multi-tenant apps
- Handling tenant-scoped API requests
- Migrating single-tenant to multi-tenant schema

## Isolation Strategies

| Strategy | Isolation | Cost | When to Use |
|----------|-----------|------|-------------|
| **Separate DB** | Strongest | Highest | Regulated industries, enterprise, high PII |
| **Separate Schema** | Strong | Medium | Mid-market SaaS, per-tenant customization |
| **Shared Schema + RLS** | Good | Lowest | SMB SaaS, cost-sensitive, similar tenant size |

This skill focuses on **Shared Schema + Row-Level Security** — the most common pattern for scalable SaaS.

## SQLAlchemy — Tenant-Aware Models

### Base Model with tenant_id

```python
from sqlalchemy import Column, String, DateTime, Index
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
import uuid

class Base(DeclarativeBase):
    pass

class TenantMixin:
    """Add to every tenant-scoped model."""
    tenant_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )

class Conversation(TenantMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Composite index — always query with tenant_id first
    __table_args__ = (
        Index("ix_conversations_tenant_user", "tenant_id", "user_id"),
    )
```

### Tenant-Scoped CRUD

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

class ConversationRepository:
    def __init__(self, db: AsyncSession, tenant_id: str):
        self.db = db
        self.tenant_id = tenant_id

    async def get(self, conversation_id: uuid.UUID) -> Conversation | None:
        result = await self.db.execute(
            select(Conversation).where(
                Conversation.tenant_id == self.tenant_id,  # ALWAYS filter by tenant
                Conversation.id == conversation_id,
            )
        )
        return result.scalar_one_or_none()

    async def list(self, user_id: str | None = None, limit: int = 50) -> list[Conversation]:
        q = select(Conversation).where(Conversation.tenant_id == self.tenant_id)
        if user_id:
            q = q.where(Conversation.user_id == user_id)
        q = q.order_by(Conversation.created_at.desc()).limit(limit)
        result = await self.db.execute(q)
        return list(result.scalars())

    async def create(self, **kwargs) -> Conversation:
        obj = Conversation(tenant_id=self.tenant_id, **kwargs)
        self.db.add(obj)
        await self.db.flush()
        return obj
```

## PostgreSQL Row-Level Security

RLS is the defense-in-depth layer — if application code forgets to filter by tenant, the DB rejects the query.

### Enable RLS on Tables

```sql
-- Enable RLS
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations FORCE ROW LEVEL SECURITY;  -- Applies to table owner too

-- Policy: users can only see their tenant's rows
CREATE POLICY tenant_isolation ON conversations
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Allow service role to bypass (for migrations, admin tools)
-- Do NOT grant BYPASSRLS to the app user
```

### Set Tenant Context per Transaction

```python
from sqlalchemy.ext.asyncio import AsyncSession

async def set_tenant_context(db: AsyncSession, tenant_id: str) -> None:
    """Call at the start of every request that touches tenant data."""
    await db.execute(
        text("SET LOCAL app.current_tenant_id = :tenant_id"),
        {"tenant_id": tenant_id},
    )

# In FastAPI dependency
async def get_tenant_db(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncSession:
    tenant_id = request.state.tenant_id  # Set by auth middleware
    await set_tenant_context(db, tenant_id)
    return db
```

## FastAPI — Tenant Context Extraction

### From JWT Claims

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer = HTTPBearer()

async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> str:
    token = credentials.credentials
    try:
        payload = decode_jwt(token)  # Your JWT decode logic
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            raise HTTPException(status_code=403, detail="No tenant in token")
        return tenant_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### From Subdomain

```python
@app.middleware("http")
async def extract_tenant_from_subdomain(request: Request, call_next):
    host = request.headers.get("host", "")
    # e.g., acme.app.example.com → tenant_id = "acme"
    subdomain = host.split(".")[0]

    tenant = await get_tenant_by_subdomain(subdomain)
    if tenant is None and request.url.path not in PUBLIC_PATHS:
        return JSONResponse({"error": "Tenant not found"}, status_code=404)

    request.state.tenant_id = tenant.id if tenant else None
    return await call_next(request)
```

## RBAC

### Roles and Permissions Model

```python
from enum import StrEnum

class TenantRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class Permission(StrEnum):
    READ_CONVERSATIONS = "conversations:read"
    WRITE_CONVERSATIONS = "conversations:write"
    MANAGE_USERS = "users:manage"
    MANAGE_BILLING = "billing:manage"
    MANAGE_SETTINGS = "settings:manage"

ROLE_PERMISSIONS: dict[TenantRole, set[Permission]] = {
    TenantRole.OWNER: set(Permission),  # All permissions
    TenantRole.ADMIN: {
        Permission.READ_CONVERSATIONS,
        Permission.WRITE_CONVERSATIONS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SETTINGS,
    },
    TenantRole.MEMBER: {
        Permission.READ_CONVERSATIONS,
        Permission.WRITE_CONVERSATIONS,
    },
    TenantRole.VIEWER: {Permission.READ_CONVERSATIONS},
}

def has_permission(role: TenantRole, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())
```

### Permission Dependency

```python
from functools import partial

def require_permission(permission: Permission):
    async def _check(
        tenant_id: str = Depends(get_current_tenant),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_tenant_db),
    ):
        membership = await get_tenant_membership(db, tenant_id, current_user.id)
        if not membership or not has_permission(membership.role, permission):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return membership
    return _check

# Usage in router
@router.delete("/conversations/{id}")
async def delete_conversation(
    id: uuid.UUID,
    _: TenantMembership = Depends(require_permission(Permission.WRITE_CONVERSATIONS)),
    db: AsyncSession = Depends(get_tenant_db),
):
    ...
```

## Tenant Model and Onboarding

```python
class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)  # subdomain
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(String(50), default="free")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)

class TenantMembership(Base):
    __tablename__ = "tenant_memberships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default=TenantRole.MEMBER)
    invited_by: Mapped[str | None] = mapped_column(String(255))
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id"),
    )
```

## Alembic Migration Patterns

### Add tenant_id to Existing Table

```python
# alembic/versions/0010_add_tenant_id_to_conversations.py

def upgrade() -> None:
    # Add column nullable first (existing rows)
    op.add_column("conversations", sa.Column("tenant_id", sa.String(36), nullable=True))

    # Backfill — set a default tenant for existing data
    op.execute("UPDATE conversations SET tenant_id = 'legacy-tenant' WHERE tenant_id IS NULL")

    # Now enforce NOT NULL
    op.alter_column("conversations", "tenant_id", nullable=False)

    # Add index
    op.create_index("ix_conversations_tenant_id", "conversations", ["tenant_id"])

    # Enable RLS (run as superuser or with elevated privileges)
    op.execute("ALTER TABLE conversations ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON conversations
        USING (tenant_id = current_setting('app.current_tenant_id', true))
    """)

def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON conversations")
    op.execute("ALTER TABLE conversations DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_conversations_tenant_id")
    op.drop_column("conversations", "tenant_id")
```

## Cross-Tenant Data Access (Admin/Internal)

```python
# For admin endpoints that need to bypass tenant isolation
async def get_admin_db(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),  # Internal admin only
) -> AsyncSession:
    # Don't set tenant context — allow full table access
    # This DB session should NEVER be passed to tenant-scoped repositories
    return db
```

## Security Checklist

- [ ] Every tenant-scoped query filters by `tenant_id` first
- [ ] RLS enabled and forced on all tenant data tables
- [ ] JWT/session contains `tenant_id` — verified server-side
- [ ] Admin endpoints are separate from tenant endpoints, protected by internal auth
- [ ] Tenant ID never comes from request body — only from authenticated token/session
- [ ] Cross-tenant data access audited and logged
- [ ] Tenant offboarding: hard-delete or anonymize data per contract/GDPR

## Reference Skills

- FastAPI dependency injection → skill: `fastapi-patterns`
- PostgreSQL indexes and RLS → skill: `postgres-patterns`
- JWT authentication → skill: `firebase-auth`
- Database migrations → skill: `database-migrations`
