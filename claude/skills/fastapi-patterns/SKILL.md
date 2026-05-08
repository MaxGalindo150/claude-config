---
name: fastapi-patterns
description: FastAPI architecture patterns, async SQLAlchemy ORM, Pydantic schemas, dependency injection, service layer, background tasks, and production-grade API design.
origin: local
---

# FastAPI Development Patterns

Production-grade FastAPI architecture patterns with async SQLAlchemy and Pydantic v2.

## When to Activate

- Building FastAPI applications
- Designing REST APIs with FastAPI + SQLAlchemy
- Working with async database sessions
- Setting up project structure and dependency injection
- Implementing service layers, background tasks, pagination

## Project Structure

### Recommended Layout

```
myproject/
├── app/
│   ├── __init__.py
│   ├── main.py              # App factory, lifespan, router registration
│   ├── config.py            # Settings via pydantic-settings
│   ├── database.py          # Async engine, session factory
│   ├── dependencies.py      # Shared Depends (db, current_user, etc.)
│   ├── models/              # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── user.py
│   │   └── product.py
│   ├── schemas/             # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── product.py
│   ├── routers/             # APIRouter per domain
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── products.py
│   └── services/            # Business logic
│       ├── __init__.py
│       ├── user_service.py
│       └── product_service.py
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── tests/
├── alembic.ini
├── pyproject.toml
└── .env
```

### App Factory with Lifespan

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.routers import users, products


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(users.router, prefix="/api/v1")
    app.include_router(products.router, prefix="/api/v1")

    return app


app = create_app()
```

### Settings with pydantic-settings

```python
# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "MyApp"
    app_version: str = "0.1.0"
    debug: bool = False

    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

## Database Setup (Async SQLAlchemy)

```python
# app/database.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass
```

## Model Design Patterns

### SQLAlchemy Async Models

```python
# app/models/base.py
from datetime import datetime
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# app/models/user.py
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    products: Mapped[list["Product"]] = relationship(back_populates="owner", lazy="selectin")


# app/models/product.py
from decimal import Decimal
from sqlalchemy import String, Text, Numeric, Integer, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin


class Product(Base, TimestampMixin):
    __tablename__ = "products"
    __table_args__ = (
        Index("ix_products_owner_active", "owner_id", "is_active"),
        CheckConstraint("price >= 0", name="ck_price_non_negative"),
        CheckConstraint("stock >= 0", name="ck_stock_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    stock: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(default=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    owner: Mapped["User"] = relationship(back_populates="products")
```

## Pydantic Schema Patterns

```python
# app/schemas/product.py
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    price: Decimal = Field(..., ge=0, decimal_places=2)
    stock: int = Field(..., ge=0)


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    price: Decimal | None = Field(None, ge=0, decimal_places=2)
    stock: int | None = Field(None, ge=0)
    is_active: bool | None = None


class ProductResponse(ProductBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    owner_id: int
    created_at: datetime
    updated_at: datetime


class ProductListResponse(BaseModel):
    items: list[ProductResponse]
    total: int
    page: int
    size: int
    pages: int
```

## Dependency Injection

```python
# app/dependencies.py
from typing import Annotated, AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSession = Annotated[AsyncSession, Depends(get_db)]
```

## Router Patterns

```python
# app/routers/products.py
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DbSession
from app.schemas.product import ProductCreate, ProductUpdate, ProductResponse, ProductListResponse
from app.services.product_service import ProductService
from app.routers.users import CurrentUser

router = APIRouter(prefix="/products", tags=["products"])


@router.get("/", response_model=ProductListResponse)
async def list_products(
    db: DbSession,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
):
    service = ProductService(db)
    return await service.list(page=page, size=size, search=search)


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: DbSession):
    service = ProductService(db)
    product = await service.get_or_404(product_id)
    return product


@router.post("/", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    data: ProductCreate,
    db: DbSession,
    current_user: CurrentUser,
):
    service = ProductService(db)
    return await service.create(data, owner_id=current_user.id)


@router.patch("/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    data: ProductUpdate,
    db: DbSession,
    current_user: CurrentUser,
):
    service = ProductService(db)
    product = await service.get_or_404(product_id)
    if product.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return await service.update(product, data)


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    product_id: int,
    db: DbSession,
    current_user: CurrentUser,
):
    service = ProductService(db)
    product = await service.get_or_404(product_id)
    if product.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    await service.delete(product)
```

## Service Layer Pattern

```python
# app/services/product_service.py
import math
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from app.models.product import Product
from app.schemas.product import ProductCreate, ProductUpdate, ProductListResponse


class ProductService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_or_404(self, product_id: int) -> Product:
        result = await self.db.get(Product, product_id)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
        return result

    async def list(
        self,
        page: int = 1,
        size: int = 20,
        search: str | None = None,
    ) -> ProductListResponse:
        query = select(Product).where(Product.is_active == True)

        if search:
            query = query.where(
                or_(
                    Product.name.ilike(f"%{search}%"),
                    Product.description.ilike(f"%{search}%"),
                )
            )

        total_result = await self.db.execute(select(func.count()).select_from(query.subquery()))
        total = total_result.scalar_one()

        query = query.offset((page - 1) * size).limit(size)
        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return ProductListResponse(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=math.ceil(total / size) if total else 0,
        )

    async def create(self, data: ProductCreate, owner_id: int) -> Product:
        product = Product(**data.model_dump(), owner_id=owner_id)
        self.db.add(product)
        await self.db.flush()
        await self.db.refresh(product)
        return product

    async def update(self, product: Product, data: ProductUpdate) -> Product:
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(product, field, value)
        await self.db.flush()
        await self.db.refresh(product)
        return product

    async def delete(self, product: Product) -> None:
        await self.db.delete(product)
        await self.db.flush()
```

## Background Tasks

```python
# app/routers/orders.py
from fastapi import APIRouter, BackgroundTasks
from app.services.email_service import send_order_confirmation

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("/", status_code=201)
async def create_order(
    data: OrderCreate,
    background_tasks: BackgroundTasks,
    db: DbSession,
    current_user: CurrentUser,
):
    service = OrderService(db)
    order = await service.create(data, user_id=current_user.id)
    background_tasks.add_task(send_order_confirmation, order.id, current_user.email)
    return order
```

## Alembic Setup

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.config import settings
from app.database import Base
# Import all models so Alembic can detect them
import app.models  # noqa: F401

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

## Error Handling

```python
# app/main.py (add to create_app)
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    return JSONResponse(
        status_code=409,
        content={"detail": "Resource already exists or constraint violation"},
    )
```

## Quick Reference

| Pattern | Description |
|---------|-------------|
| `AsyncSession` | Always use async DB session |
| `Depends(get_db)` | Inject DB session, auto-commit/rollback |
| `Annotated[X, Depends(...)]` | Reusable typed dependency aliases |
| `mapped_column` | Modern SQLAlchemy 2.x column definition |
| `model_dump(exclude_unset=True)` | Partial updates — only set fields |
| `expire_on_commit=False` | Keep objects accessible after commit |
| `lazy="selectin"` | Async-safe relationship loading |
| `lru_cache` on settings | Load env vars once |
| `lifespan` | Startup/shutdown events |
| `BackgroundTasks` | Fire-and-forget async work |

Build for explicitness: FastAPI's DI system is powerful, but keep dependencies focused and services stateless (except for the db session).
