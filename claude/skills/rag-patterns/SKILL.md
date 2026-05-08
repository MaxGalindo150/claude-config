---
name: rag-patterns
description: RAG architecture patterns — document ingestion, chunking strategies, embedding models, vector stores, hybrid retrieval, re-ranking, advanced techniques (HyDE, multi-query), and evaluation with RAGAS.
origin: local
---

# RAG Architecture Patterns

Production-grade Retrieval-Augmented Generation patterns from ingestion to evaluation.

## When to Activate

- Designing or reviewing a RAG pipeline
- Choosing a chunking strategy, embedding model, or vector store
- Implementing hybrid retrieval or re-ranking
- Setting up RAG evaluation with RAGAS
- Debugging retrieval quality issues

## RAG Pipeline Overview

```
Documents → Ingestion → Chunking → Embedding → Vector Store
                                                     ↓
Query → Embedding → Retrieval → Re-ranking → Context Assembly → LLM → Answer
```

Each step is independently tunable. Start simple; add complexity only when evals show it's needed.

## Document Ingestion

### Loaders by Source

```python
# Use langchain-community loaders or write your own
from langchain_community.document_loaders import (
    PyPDFLoader,          # PDFs
    UnstructuredMarkdownLoader,  # Markdown
    WebBaseLoader,        # Web pages
    DirectoryLoader,      # Batch local files
)

# Custom loader pattern
from langchain_core.documents import Document

class CustomLoader:
    def __init__(self, source: str) -> None:
        self.source = source

    def load(self) -> list[Document]:
        # Always attach rich metadata — crucial for filtering later
        return [
            Document(
                page_content=text,
                metadata={
                    "source": self.source,
                    "doc_id": doc_id,
                    "created_at": iso_date,
                    "type": "contract",   # domain-specific
                },
            )
        ]
```

### Metadata Strategy

Good metadata enables **pre-filtering** before vector search — far cheaper and more precise than relying on embedding similarity alone.

| Metadata field | Purpose |
|----------------|---------|
| `source` / `url` | Attribution, deduplication |
| `doc_id` | Parent-document retrieval |
| `section` / `heading` | Hierarchical filtering |
| `created_at` | Recency filtering |
| `language` | Multi-lingual routing |
| `type` / `category` | Domain filtering |

## Chunking Strategies

### Choosing a Strategy

| Strategy | Best for | Avoid when |
|----------|----------|------------|
| Fixed-size | Quick prototypes, homogeneous text | Structured docs (code, tables) |
| Recursive | General prose | Complex nested structures |
| Semantic | Conceptually dense docs | Cost-sensitive pipelines |
| Markdown-aware | Docs/wikis with headers | Docs without headings |
| Late chunking | Long docs with global context | Very short documents |

### Recursive Character Splitter (default choice)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # tokens ≈ chars / 4; tune per embedding model
    chunk_overlap=64,     # ~10-15% overlap preserves context across boundaries
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)
```

### Markdown-Aware Splitter

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# Each chunk inherits heading metadata — enables section-level filtering
```

### Semantic Chunking

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Splits on embedding distance spikes — respects topic boundaries
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95,
)
chunks = splitter.split_documents(documents)
```

### Chunking Rules of Thumb

- Chunk size should fit comfortably in the embedding model's token limit (typically 512 tokens)
- Overlap of ~10% prevents context loss at boundaries
- Always propagate metadata from parent document to all child chunks
- Keep code blocks and tables intact — never split mid-structure
- Measure retrieval quality before optimizing chunk size

## Embedding Models

### Selection Guide

| Model | Dims | Context | Notes |
|-------|------|---------|-------|
| `text-embedding-3-small` | 1536 | 8191 | Fast, cheap, good default |
| `text-embedding-3-large` | 3072 | 8191 | Best OpenAI quality |
| `cohere-embed-v3` | 1024 | 512 | Excellent multilingual |
| `BAAI/bge-m3` | 1024 | 8192 | Best open-source, multilingual |
| `nomic-embed-text` | 768 | 8192 | Good open-source, Apache 2.0 |

```python
# OpenAI (reliable default)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Local / open-source (no API cost, runs on-prem)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

### Embedding Best Practices

- Use the **same model** for indexing and querying — never mix
- Normalize embeddings (cosine similarity then equals dot product — faster)
- For multilingual corpora use a multilingual model; don't translate at ingestion
- Cache embeddings: re-embedding is expensive and deterministic

## Vector Stores

### Comparison

| Store | Best for | Notes |
|-------|----------|-------|
| `pgvector` | Existing PostgreSQL infra | Great for <10M vectors; no extra infra |
| `Chroma` | Local dev / prototypes | Zero config, in-process |
| `Pinecone` | Managed, large scale | Expensive; no self-hosting |
| `Qdrant` | Self-hosted production | Fast, rich filtering, Docker-friendly |
| `Weaviate` | Multi-modal, hybrid search | More complex ops |

### pgvector (recommended for most FastAPI projects)

```python
from langchain_postgres import PGVector
from sqlalchemy import create_engine

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="docs",
    connection=settings.database_url,
    use_jsonb=True,   # enables metadata filtering
)

# Ingest
vector_store.add_documents(chunks)

# Query with metadata filter
results = vector_store.similarity_search(
    query="What is the refund policy?",
    k=5,
    filter={"type": "contract", "language": "en"},
)
```

### Chroma (local dev)

```python
import chromadb
from langchain_chroma import Chroma

client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(
    client=client,
    collection_name="docs",
    embedding_function=embeddings,
)
```

## Retrieval Strategies

### 1. Semantic Search (baseline)

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
```

### 2. Hybrid Search (BM25 + vector) — usually better than semantic alone

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],   # tune based on RAGAS evals
)
```

### 3. MMR — Maximum Marginal Relevance (reduces redundancy)

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)
# lambda_mult: 0 = max diversity, 1 = max relevance
```

### 4. Self-Query (LLM extracts metadata filters from natural language)

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document

metadata_field_info = [
    AttributeInfo(name="type", description="Document type", type="string"),
    AttributeInfo(name="created_at", description="Creation date", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents="Legal contracts and policies",
    metadata_field_info=metadata_field_info,
)
# Query: "Find contracts from 2024" → auto-adds filter {"created_at": {"$gte": "2024-01-01"}}
```

## Re-Ranking

Re-rankers significantly improve precision. Always add after initial retrieval.

```python
# Option 1: Cohere Rerank (managed, excellent quality)
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

compressor = CohereRerank(model="rerank-english-v3.0", top_n=3)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever,
)

# Option 2: Cross-encoder (self-hosted, no API cost)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=3)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever,
)
```

## Advanced Retrieval Techniques

### HyDE — Hypothetical Document Embeddings

Generate a hypothetical answer, embed it, search. Improves recall for questions with sparse exact-match keywords.

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    custom_prompt=PromptTemplate(
        input_variables=["question"],
        template="Write a detailed answer to: {question}\n\nAnswer:",
    ),
)
hyde_store = vector_store.__class__(embedding_function=hyde_embeddings, ...)
```

### Multi-Query Retrieval

Generate N reformulations of the query, retrieve for each, deduplicate.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever,
    llm=llm,
)
# Generates 3 alternative queries internally, unions results
```

### Parent-Document Retriever

Index small chunks for precise retrieval, return full parent sections to the LLM.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=InMemoryStore(),   # use Redis/DB in production
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)
```

## Context Assembly

```python
def format_context(docs: list[Document], max_tokens: int = 4000) -> str:
    """Assemble retrieved docs into a context string, respecting token budget."""
    context_parts = []
    total_chars = 0
    char_limit = max_tokens * 4  # rough chars-per-token estimate

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        text = doc.page_content.strip()

        if total_chars + len(text) > char_limit:
            break

        context_parts.append(f"[{i}] Source: {source}\n{text}")
        total_chars += len(text)

    return "\n\n---\n\n".join(context_parts)
```

## RAG Evaluation with RAGAS

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Answer grounded in context?
    answer_relevancy,      # Answer relevant to question?
    context_precision,     # Retrieved context precise?
    context_recall,        # Retrieved context complete?
)
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,      # list[list[str]]
    "ground_truth": reference_answers,  # for context_recall
})

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
# {'faithfulness': 0.87, 'answer_relevancy': 0.91, ...}
```

### RAGAS Score Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Faithfulness | 0.80 | 0.90+ |
| Answer Relevancy | 0.80 | 0.90+ |
| Context Precision | 0.70 | 0.85+ |
| Context Recall | 0.70 | 0.85+ |

### Debugging Low Scores

| Low metric | Likely cause | Fix |
|------------|-------------|-----|
| Faithfulness | LLM hallucinating beyond context | Tighten system prompt, lower temperature |
| Answer Relevancy | Vague or off-topic answers | Improve prompt instructions |
| Context Precision | Noisy retrieval | Add re-ranker, improve metadata filters |
| Context Recall | Missing relevant chunks | More overlap, larger k, hybrid search |

## Common Anti-Patterns

| Anti-pattern | Problem | Fix |
|--------------|---------|-----|
| Chunk size too large (>1024 tokens) | Dilutes signal; embedding captures too much | Use 256–512 tokens |
| No metadata on chunks | Can't filter; retrieves unrelated docs | Always carry source/type/date |
| Only semantic search | Misses exact keyword matches | Add BM25 (hybrid) |
| No re-ranker | Top-k noisy; first retrieved ≠ most relevant | Add cross-encoder |
| No eval dataset | Can't measure improvements | Build golden set early |
| Embedding model mismatch | Index and query in different spaces | Same model always |
| Stuffing all k chunks | Context overload; LLM loses focus | Re-rank to top 3 |

## Quick Reference

| Decision | Default choice |
|----------|---------------|
| Chunking | `RecursiveCharacterTextSplitter(512, overlap=64)` |
| Embedding | `text-embedding-3-small` or `BAAI/bge-m3` (local) |
| Vector store | `pgvector` (existing PG) / `Qdrant` (new infra) |
| Retrieval | Hybrid (BM25 + vector, 40/60) |
| Re-ranking | Cohere Rerank v3 or `bge-reranker-v2-m3` |
| Eval | RAGAS (faithfulness + context_precision minimum) |

Start with the simplest pipeline that passes your eval. Optimize one component at a time.
