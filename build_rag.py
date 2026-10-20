#!/usr/bin/env python3
"""
build_rag.py
Builds a persistent local ChromaDB vector index over data/emails_corpus.jsonl
using BAAI/bge-small-en-v1.5 embeddings so the local model can retrieve exact dates,
senders, threads, and facts from your emails.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


def chunk_email(record: Dict, chunk_size: int = 1200, overlap: int = 200) -> List[Tuple[str, str, Dict]]:
    header_prefix = (
        f"Date: {record.get('date', '')}\n"
        f"From: {record.get('from', '')}\n"
        f"To: {record.get('to', '')}\n"
        f"Subject: {record.get('subject', '')}\n\n"
    )
    body = record.get("body", "")
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(body):
        end = min(len(body), start + chunk_size)
        snippet = body[start:end]
        doc_text = header_prefix + snippet
        chunk_id = f"{record['id']}_chunk_{chunk_idx}"
        metadata = {
            "email_id": record["id"],
            "date": record.get("date", "")[:100],
            "from": record.get("from", "")[:150],
            "to": record.get("to", "")[:150],
            "subject": record.get("subject", "")[:200],
            "is_sent": bool(record.get("is_sent", False)),
        }
        chunks.append((chunk_id, doc_text, metadata))
        if end == len(body):
            break
        start = max(0, end - overlap)
        chunk_idx += 1
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index parsed Gmail corpus into a local ChromaDB vector store.")
    parser.add_argument("--corpus", default="data/emails_corpus.jsonl", help="Path to emails_corpus.jsonl")
    parser.add_argument("--db-dir", default="chroma_db", help="Directory for persistent ChromaDB storage")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Local SentenceTransformer model for embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding and insertion")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}. Run parse_mbox.py first.")

    print(f"[build_rag] Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)

    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.get_or_create_collection(
        name="personal_emails",
        metadata={"hnsw:space": "cosine"},
    )

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[Dict] = []

    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for cid, doc_text, meta in chunk_email(record):
                all_ids.append(cid)
                all_docs.append(doc_text)
                all_metas.append(meta)

    print(f"[build_rag] Indexing {len(all_docs)} email chunks into {args.db_dir}...")
    for i in range(0, len(all_docs), args.batch_size):
        batch_ids = all_ids[i : i + args.batch_size]
        batch_docs = all_docs[i : i + args.batch_size]
        batch_metas = all_metas[i : i + args.batch_size]
        embeddings = embedder.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )

    print(f"[build_rag] Successfully indexed {len(all_docs)} chunks in collection 'personal_emails'.")


if __name__ == "__main__":
    main()
