#!/usr/bin/env python3
"""
rag_engine.py
Modular ChromaDB Vector Retrieval Engine:
  - Supports dynamic Attach / Detach on demand.
  - Recency-aware searching: parses RFC 2822 timestamps so the newest emails take priority for career & current info.
  - Incremental indexing: when re-exporting in the future, only newly added emails are embedded.
  - Data distillation: extracts salient facts, dates, and subjects instead of dumping raw junk.
"""

import email.utils
import json
import os
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


def parse_email_timestamp(date_str: str) -> Tuple[float, str]:
    """Converts RFC 2822 date to (unix_timestamp, iso_date_string)."""
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        return dt.timestamp(), dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return 0.0, "Unknown Date"


class RAGEngine:
    def __init__(
        self,
        db_dir: str = "chroma_db",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        attached: bool = True,
    ):
        self.db_dir = db_dir
        self.embedding_model_name = embedding_model
        self.attached = attached
        self._embedder: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    @property
    def is_attached(self) -> bool:
        return self.attached and os.path.exists(self.db_dir)

    def attach(self) -> str:
        self.attached = True
        return f"ChromaDB attached ({self.count_chunks()} indexed email chunks)."

    def detach(self) -> str:
        self.attached = False
        return "ChromaDB detached. Assistant will answer using working memory and model intelligence."

    def toggle(self) -> str:
        if self.attached:
            return self.detach()
        return self.attach()

    def _ensure_loaded(self) -> None:
        if not self.is_attached:
            return
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_dir)
            self._collection = self._client.get_or_create_collection(
                name="personal_emails",
                metadata={"hnsw:space": "cosine"},
            )

    def count_chunks(self) -> int:
        if not os.path.exists(self.db_dir):
            return 0
        try:
            client = chromadb.PersistentClient(path=self.db_dir)
            coll = client.get_collection(name="personal_emails")
            return coll.count()
        except Exception:
            return 0

    def search(
        self,
        query: str,
        top_k: int = 4,
        prefer_recent: bool = False,
    ) -> List[Dict]:
        """
        Retrieves relevant email chunks.
        If prefer_recent is True (e.g. for company, salary, recent events),
        fetches extra candidates and sorts by email timestamp descending.
        """
        if not self.is_attached:
            return []

        self._ensure_loaded()
        q_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()
        fetch_k = top_k * 3 if prefer_recent else top_k
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=fetch_k,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)

        candidates = []
        for doc, meta, dist in zip(docs, metas, distances):
            date_raw = meta.get("date", "")
            ts = meta.get("timestamp")
            if ts is None:
                ts, _ = parse_email_timestamp(date_raw)
            else:
                ts = float(ts)

            candidates.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "timestamp": ts,
                "date": date_raw,
                "subject": meta.get("subject", ""),
                "from": meta.get("from", ""),
                "to": meta.get("to", ""),
            })

        if prefer_recent:
            candidates.sort(key=lambda x: x["timestamp"], reverse=True)

        return candidates[:top_k]

    def format_context(self, snippets: List[Dict]) -> str:
        """Formats extracted snippets concisely for the LLM."""
        if not snippets:
            return "No relevant emails retrieved."
        formatted = []
        for i, s in enumerate(snippets, 1):
            formatted.append(
                f"[Email {i}] Date: {s['date']} | From: {s['from']} | Subject: {s['subject']}\n"
                f"{s['document']}"
            )
        return "\n\n---\n\n".join(formatted)

    def search_and_format(self, query: str, top_k: int = 4) -> str:
        if not self.is_attached:
            return ""

        query_lower = query.lower()
        prefer_recent = any(
            kw in query_lower
            for kw in [
                "current company",
                "latest company",
                "latest role",
                "where do i work",
                "where does abhishek work",
                "current ctc",
                "latest ctc",
                "recent",
                "last email",
                "latest email",
                "current organization",
            ]
        )
        snippets = self.search(query, top_k=top_k, prefer_recent=prefer_recent)
        return self.format_context(snippets)
