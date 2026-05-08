---
name: fastapi-tdd
description: FastAPI testing with TDD — pytest-asyncio, httpx AsyncClient, async SQLAlchemy fixtures, polyfactory, mocking external services, and coverage targets.
origin: local
---

# FastAPI Testing with TDD

Test-driven development for FastAPI applications using pytest-asyncio, httpx, and async SQLAlchemy.

## When to Activate

- Writing new FastAPI endpoints or services
- Setting up testing infrastructure for a FastAPI project
- Testing async routes, dependencies, and database interactions
- Implementing TDD on FastAPI + Alembic + SQLAlchemy projects

## TDD Workflow

### Red-Green-Refactor for FastAPI

```python
# Step 1: RED — Write failing test first
async def test_create_product_returns_201(client, auth_headers):
    response = await client.post(
        "/api/v1/products/",
        json={"name": "Widget", "price": "9.99", "stock": 100},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert data["id"] is not None

# Step 2: GREEN — Implement the route + service to make it pass
# Step 3: REFACTOR — Improve while keeping tests green
```

## Setup

### Dependencies

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=app",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
    "--strict-markers",
]
markers = [
    "slow: marks tests as slow (deselect with -m 'not slow')",
    "integration: marks integration tests",
]

[tool.coverage.run]
omit = ["tests/*", "alembic/*", "app/main.py"]
```

```bash
# Install test dependencies
uv add --dev pytest pytest-asyncio httpx pytest-cov anyio polyfactory
```

### Async Test Database

```python
# tests/conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from app.database import Base, get_db

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestAsyncSessionLocal = async_sessionmaker(
    bind=test_engine, expire_on_commit=False, autoflush=False
)


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture
async def db() -> AsyncSession:
    async with TestAsyncSessionLocal() as session:
        yield session
        await session.rollback()  # Isolate each test


@pytest.fixture
def override_db(db: AsyncSession):
    """Override the app's get_db dependency with the test session."""
    async def _override():
        yield db

    app.dependency_overrides[get_db] = _override
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client(override_db) -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
```

### User + Auth Fixtures

```python
# tests/conftest.py (continued)
import pytest
from app.auth.passwords import hash_password
from app.auth.tokens import create_access_token
from app.models.user import User


@pytest.fixture
async def user(db: AsyncSession) -> User:
    u = User(
        email="test@example.com",
        hashed_password=hash_password("TestPass123"),
        is_active=True,
    )
    db.add(u)
    await db.flush()
    await db.refresh(u)
    return u


@pytest.fixture
async def admin_user(db: AsyncSession) -> User:
    u = User(
        email="admin@example.com",
        hashed_password=hash_password("AdminPass123"),
        is_active=True,
        is_superuser=True,
        role="admin",
    )
    db.add(u)
    await db.flush()
    await db.refresh(u)
    return u


@pytest.fixture
def auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(subject=user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(admin_user: User) -> dict[str, str]:
    token = create_access_token(subject=admin_user.id)
    return {"Authorization": f"Bearer {token}"}
```

## Polyfactory for Test Data

```python
# tests/factories.py
from decimal import Decimal
from polyfactory.factories.sqlalchemy_factory import SQLAlchemyFactory
from app.models.user import User
from app.models.product import Product


class UserFactory(SQLAlchemyFactory):
    __model__ = User
    __set_relationships__ = False

    email = SQLAlchemyFactory.faker.email
    hashed_password = "$2b$12$fakehash"  # pre-hashed placeholder
    is_active = True
    is_superuser = False


class ProductFactory(SQLAlchemyFactory):
    __model__ = Product
    __set_relationships__ = False

    name = SQLAlchemyFactory.faker.sentence(nb_words=3)
    description = SQLAlchemyFactory.faker.text(max_nb_chars=200)
    price = Decimal("49.99")
    stock = 50
    is_active = True


# Usage in tests
async def test_something(db):
    product = ProductFactory.build()          # in-memory, no DB
    product = await ProductFactory.create_async(db)  # persisted
    products = await ProductFactory.create_batch_async(5, db)
```

## Model / Service Testing

```python
# tests/test_services/test_product_service.py
import pytest
from decimal import Decimal
from fastapi import HTTPException

from app.services.product_service import ProductService
from app.schemas.product import ProductCreate, ProductUpdate
from tests.factories import ProductFactory, UserFactory


class TestProductService:

    async def test_create_product(self, db, user):
        service = ProductService(db)
        data = ProductCreate(name="Widget", price=Decimal("9.99"), stock=10)

        product = await service.create(data, owner_id=user.id)

        assert product.id is not None
        assert product.name == "Widget"
        assert product.owner_id == user.id

    async def test_get_or_404_raises_for_missing(self, db):
        service = ProductService(db)

        with pytest.raises(HTTPException) as exc_info:
            await service.get_or_404(99999)

        assert exc_info.value.status_code == 404

    async def test_update_partial(self, db, user):
        product = await ProductFactory.create_async(db, owner_id=user.id)
        service = ProductService(db)
        data = ProductUpdate(name="Updated Name")

        updated = await service.update(product, data)

        assert updated.name == "Updated Name"
        assert updated.price == product.price  # unchanged

    async def test_list_filters_inactive(self, db):
        await ProductFactory.create_batch_async(3, db, is_active=True)
        await ProductFactory.create_batch_async(2, db, is_active=False)
        service = ProductService(db)

        result = await service.list()

        assert result.total == 3

    async def test_list_search(self, db):
        await ProductFactory.create_async(db, name="Apple iPhone", is_active=True)
        await ProductFactory.create_async(db, name="Samsung Galaxy", is_active=True)
        service = ProductService(db)

        result = await service.list(search="Apple")

        assert result.total == 1
        assert result.items[0].name == "Apple iPhone"

    async def test_list_pagination(self, db):
        await ProductFactory.create_batch_async(15, db, is_active=True)
        service = ProductService(db)

        page1 = await service.list(page=1, size=10)
        page2 = await service.list(page=2, size=10)

        assert len(page1.items) == 10
        assert len(page2.items) == 5
        assert page1.pages == 2
```

## API Endpoint Testing

```python
# tests/test_routers/test_products.py
import pytest
from httpx import AsyncClient


class TestProductEndpoints:

    async def test_list_products_unauthenticated(self, client: AsyncClient):
        response = await client.get("/api/v1/products/")
        assert response.status_code == 200  # public endpoint

    async def test_create_product_requires_auth(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/products/",
            json={"name": "Widget", "price": "9.99", "stock": 10},
        )
        assert response.status_code == 401

    async def test_create_product(self, client: AsyncClient, auth_headers):
        response = await client.post(
            "/api/v1/products/",
            json={"name": "Widget", "price": "9.99", "stock": 10},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Widget"
        assert "id" in data

    async def test_create_product_invalid_price(self, client: AsyncClient, auth_headers):
        response = await client.post(
            "/api/v1/products/",
            json={"name": "Widget", "price": "-1.00", "stock": 10},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_get_product_not_found(self, client: AsyncClient):
        response = await client.get("/api/v1/products/99999")
        assert response.status_code == 404

    async def test_update_product_forbidden_for_non_owner(
        self, client: AsyncClient, auth_headers, db
    ):
        other_user = await UserFactory.create_async(db)
        product = await ProductFactory.create_async(db, owner_id=other_user.id)

        response = await client.patch(
            f"/api/v1/products/{product.id}",
            json={"name": "Hijacked"},
            headers=auth_headers,
        )
        assert response.status_code == 403

    async def test_delete_product_owner_can_delete(
        self, client: AsyncClient, auth_headers, user, db
    ):
        product = await ProductFactory.create_async(db, owner_id=user.id)

        response = await client.delete(
            f"/api/v1/products/{product.id}", headers=auth_headers
        )
        assert response.status_code == 204

    async def test_list_pagination_response_shape(self, client: AsyncClient):
        response = await client.get("/api/v1/products/?page=1&size=10")
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "pages" in data
```

## Auth Endpoint Testing

```python
# tests/test_routers/test_auth.py
class TestAuth:

    async def test_login_success(self, client: AsyncClient, user):
        response = await client.post(
            "/api/v1/auth/token",
            data={"username": user.email, "password": "TestPass123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_wrong_password(self, client: AsyncClient, user):
        response = await client.post(
            "/api/v1/auth/token",
            data={"username": user.email, "password": "WrongPass"},
        )
        assert response.status_code == 401

    async def test_login_unknown_user(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/auth/token",
            data={"username": "nobody@example.com", "password": "Pass123"},
        )
        assert response.status_code == 401

    async def test_protected_endpoint_with_invalid_token(self, client: AsyncClient):
        response = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401
```

## Mocking External Services

```python
# tests/test_services/test_email.py
from unittest.mock import AsyncMock, patch
import pytest


async def test_order_confirmation_email_sent(db, user):
    with patch("app.services.email_service.send_email", new_callable=AsyncMock) as mock_send:
        from app.services.order_service import OrderService
        service = OrderService(db)
        order = await service.create(...)

        mock_send.assert_called_once_with(
            to=user.email,
            subject="Order Confirmation",
        )


# Override entire dependency for a test
async def test_with_mock_payment_gateway(client, auth_headers):
    with patch("app.routers.orders.PaymentGateway.charge") as mock_charge:
        mock_charge.return_value = {"status": "succeeded", "id": "ch_123"}

        response = await client.post("/api/v1/orders/", ..., headers=auth_headers)

        assert response.status_code == 201
        mock_charge.assert_called_once()
```

## Testing Best Practices

### DO

- **`asyncio_mode = "auto"`** — no need to mark every test `@pytest.mark.asyncio`
- **Roll back after each test** — use `await session.rollback()` in the db fixture
- **Override `get_db` via `dependency_overrides`** — inject test session into app
- **Test the HTTP contract** — status codes, response shape, error messages
- **Test authorization explicitly** — unauthenticated, wrong user, missing role
- **Use `polyfactory`** for realistic but isolated test data
- **Mock at the boundary** — mock external HTTP calls, email clients, S3, etc.

### DON'T

- **Don't use the production DB** — always a separate test DB
- **Don't share state between tests** — each test rolls back
- **Don't test Pydantic / SQLAlchemy internals** — trust the libraries
- **Don't over-mock** — real DB queries catch real bugs
- **Don't hardcode IDs** — use fixture-created objects

## Coverage Targets

| Component | Target |
|-----------|--------|
| Routers (endpoints) | 85%+ |
| Services (business logic) | 90%+ |
| Auth / security | 90%+ |
| Schemas (validators) | 80%+ |
| Overall | 80%+ |

```bash
# Run all tests with coverage
pytest

# Run only unit tests (skip slow/integration)
pytest -m "not slow and not integration"

# Run a specific file
pytest tests/test_routers/test_products.py -v

# Show which lines are missing coverage
pytest --cov=app --cov-report=term-missing
```

## Quick Reference

| Pattern | Usage |
|---------|-------|
| `asyncio_mode = "auto"` | Auto-async tests in pytest.ini |
| `ASGITransport(app=app)` | Mount FastAPI into httpx |
| `app.dependency_overrides` | Swap Depends for tests |
| `await session.rollback()` | Isolate DB state per test |
| `SQLAlchemyFactory` | Generate ORM model instances |
| `patch("module.func", AsyncMock)` | Mock async external calls |
| `auth_headers` fixture | Reuse JWT headers across tests |
| `pytest.raises(HTTPException)` | Assert service-layer errors |

Remember: Tests are the first consumer of your API. If they're painful to write, the design needs work.
