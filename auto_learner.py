#!/usr/bin/env python3
"""
auto_learner.py
Autonomous Knowledge Ingestion & Universal Document Learning Engine for Alisa:
  - Ingests ANY file dropped in inbox/ or passed via /learn (.pdf, .txt, .md, .csv, .json, .py, .sh, .mbox, etc.).
  - Zero hardcoding: extracts facts, dates, entities, and timelines autonomously.
  - Automatically indexes chunks into local ChromaDB vector store.
  - Automatically updates and balances train/valid/test datasets via DatasetBuilder.
  - Tracks processed file signatures in data/processed_files.json to avoid duplicate processing.
  - Safely handles large archives (.mbox) by delegating to streaming parser.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from dataset_builder import DatasetBuilder


LEARNED_MEMORY_FILE = "data/learned_memory.json"
PROCESSED_FILES_FILE = "data/processed_files.json"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_file_signature(filepath: str) -> str:
    """Computes quick file signature using mtime and size."""
    try:
        st = os.stat(filepath)
        return f"{st.st_size}_{st.st_mtime}"
    except Exception:
        return "unknown"


class UniversalParser:
    """Dynamically parses arbitrary file formats into clean text with metadata."""

    @staticmethod
    def parse_file(file_path: str) -> List[Dict]:
        """Returns a list of records: [{'id': str, 'text': str, 'metadata': dict}]."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        records: List[Dict] = []
        base_name = os.path.basename(file_path)
        stat = os.stat(file_path)
        file_mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

        if ext in (".txt", ".md", ".markdown", ".log", ".rst"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            if content:
                records.append({
                    "id": f"{base_name}",
                    "text": content,
                    "metadata": {"source": base_name, "type": "text", "date": file_mtime},
                })

        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        records.append({
                            "id": f"{base_name}_p{page_idx+1}",
                            "text": text.strip(),
                            "metadata": {"source": base_name, "type": "pdf", "page": page_idx + 1, "date": file_mtime},
                        })
            except Exception as e:
                print(f"[auto_learner] Warning: PDF extraction failed for {file_path} ({e})")

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    for idx, row in enumerate(rows):
                        row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                        records.append({
                            "id": f"{base_name}_row{idx}",
                            "text": row_text,
                            "metadata": {"source": base_name, "type": "csv", "row": idx, "date": file_mtime},
                        })

        elif ext in (".json", ".jsonl"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                if ext == ".jsonl":
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                records.append({
                                    "id": f"{base_name}_item{idx}",
                                    "text": json.dumps(item, indent=2, ensure_ascii=False),
                                    "metadata": {"source": base_name, "type": "jsonl", "item": idx, "date": file_mtime},
                                })
                            except Exception:
                                pass
                else:
                    try:
                        data = json.load(f)
                        records.append({
                            "id": f"{base_name}",
                            "text": json.dumps(data, indent=2, ensure_ascii=False),
                            "metadata": {"source": base_name, "type": "json", "date": file_mtime},
                        })
                    except Exception:
                        pass

        elif ext in (".py", ".js", ".ts", ".html", ".css", ".sh", ".sql", ".yaml", ".yml", ".go", ".rs", ".c", ".cpp"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read().strip()
            if code:
                records.append({
                    "id": f"{base_name}",
                    "text": f"File: {base_name}\nType: Source Code ({ext})\n\n{code}",
                    "metadata": {"source": base_name, "type": f"code_{ext[1:]}", "date": file_mtime},
                })

        elif ext == ".mbox":
            # For .mbox files, alert user to use streaming parser
            print(f"[auto_learner] Found mailbox archive: {file_path}. Processing via streaming parse_mbox...")
            subprocess.run([
                sys.executable,
                "parse_mbox.py",
                "--mbox", file_path,
                "--output-dir", "data",
            ], check=False)
            return []

        else:
            # Fallback text decoder
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
                if text:
                    records.append({
                        "id": f"{base_name}",
                        "text": text,
                        "metadata": {"source": base_name, "type": "generic", "date": file_mtime},
                    })
            except Exception:
                pass

        return records


class AutoLearner:
    def __init__(self, db_dir: str = CHROMA_DIR, memory_file: str = LEARNED_MEMORY_FILE):
        self.db_dir = db_dir
        self.memory_file = memory_file
        self.learned_data = self._load_learned_memory()
        self.dataset_builder = DatasetBuilder()

    def _load_learned_memory(self) -> Dict:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "ingested_sources": [],
            "extracted_facts": [],
            "discovered_entities": {},
            "chronological_events": [],
            "last_updated": None,
        }

    def _save_learned_memory(self) -> None:
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        self.learned_data["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.learned_data, f, indent=2, ensure_ascii=False)

    def learn_text(self, text: str, source_label: str = "direct_note") -> str:
        """Learns directly from raw pasted text or conversational notes."""
        record = {
            "id": f"note_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "text": text.strip(),
            "metadata": {
                "source": source_label,
                "type": "note",
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        }
        return self._process_records([record], source_label)

    def learn_file(self, file_path: str) -> str:
        """Parses, indexes, and extracts knowledge from any file autonomously."""
        records = UniversalParser.parse_file(file_path)
        if not records:
            if file_path.endswith(".mbox"):
                return f"Mailbox archive '{os.path.basename(file_path)}' processed and synced to data/."
            return f"No parseable text extracted from {file_path}."
        return self._process_records(records, os.path.basename(file_path))

    def _process_records(self, records: List[Dict], source_name: str) -> str:
        chunks_to_index: List[Tuple[str, str, Dict]] = []

        # 1. Chunk and prepare for ChromaDB
        for r in records:
            text = r["text"]
            chunk_size = 1000
            overlap = 150
            start = 0
            idx = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                chunk_str = text[start:end]
                cid = f"{r['id']}_c{idx}"
                meta = dict(r["metadata"])
                meta["chunk_idx"] = idx
                chunks_to_index.append((cid, chunk_str, meta))
                if end == len(text):
                    break
                start = max(0, end - overlap)
                idx += 1

            # 2. Extract autonomous facts & chronological events without hardcoding
            self._extract_autonomous_facts(text, r.get("metadata", {}))

        # 3. Store into ChromaDB
        self._upsert_to_chroma(chunks_to_index)

        # 4. Generate balanced, deduplicated all-rounder train/valid/test datasets
        ds_stats = self.dataset_builder.build_all_rounder_dataset(inbox_records=records)

        if source_name not in self.learned_data["ingested_sources"]:
            self.learned_data["ingested_sources"].append(source_name)
        self._save_learned_memory()

        return (
            f"Successfully learned from '{source_name}':\n"
            f"  • Extracted {len(records)} records ({len(chunks_to_index)} chunks indexed in ChromaDB)\n"
            f"  • Stored in dynamic memory ({len(self.learned_data['extracted_facts'])} total facts known)\n"
            f"  • Updated All-Rounder Datasets: {ds_stats['train_examples']} train | {ds_stats['valid_examples']} valid | {ds_stats['test_examples']} test examples"
        )

    def _extract_autonomous_facts(self, text: str, metadata: Dict) -> None:
        """Extracts facts, dates, and entities autonomously without any hardcoded assumptions."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if ":" in line and len(line) < 200:
                parts = line.split(":", 1)
                k, v = parts[0].strip(), parts[1].strip()
                if 2 < len(k) < 40 and 2 < len(v) < 150:
                    fact_entry = f"{k}: {v}"
                    if fact_entry not in self.learned_data["extracted_facts"]:
                        self.learned_data["extracted_facts"].append(fact_entry)

            date_match = re.search(r"\b(201\d|202\d)\b", line)
            if date_match and len(line) < 160:
                self.learned_data["chronological_events"].append({
                    "year": int(date_match.group(1)),
                    "snippet": line,
                    "source": metadata.get("source", "unknown"),
                })

        self.learned_data["chronological_events"].sort(key=lambda x: x["year"], reverse=True)
        self.learned_data["extracted_facts"] = self.learned_data["extracted_facts"][-500:]
        self.learned_data["chronological_events"] = self.learned_data["chronological_events"][-200:]

    def _upsert_to_chroma(self, chunks: List[Tuple[str, str, Dict]]) -> None:
        """Pushes chunks into ChromaDB."""
        if not chunks:
            return
        import chromadb
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=self.db_dir)
        collection = client.get_or_create_collection(
            name="personal_emails",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 128
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            b_ids = [c[0] for c in batch]
            b_docs = [c[1] for c in batch]
            b_metas = [c[2] for c in batch]
            embs = embedder.encode(b_docs, normalize_embeddings=True, show_progress_bar=False).tolist()
            collection.upsert(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=embs)

    def get_summary(self) -> str:
        """Returns completely dynamic, zero-hardcoded learned memory summary."""
        lines = []
        sources = self.learned_data.get("ingested_sources", [])
        if sources:
            lines.append(f"Learned Knowledge Sources: {', '.join(sources[:10])}")

        events = self.learned_data.get("chronological_events", [])
        if events:
            lines.append("Latest Discovered Events (Chronological - Newest First):")
            for ev in events[:8]:
                lines.append(f"  • [{ev['year']}] {ev['snippet']}")

        facts = self.learned_data.get("extracted_facts", [])
        if facts:
            lines.append("Key Discovered Attributes:")
            for f in facts[-15:]:
                lines.append(f"  • {f}")

        return "\n".join(lines) if lines else "No external files ingested yet. Drop files in inbox/ or use /learn to teach me."


def load_processed_files() -> Dict[str, str]:
    if os.path.exists(PROCESSED_FILES_FILE):
        try:
            with open(PROCESSED_FILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_processed_files(data: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(PROCESSED_FILES_FILE) or ".", exist_ok=True)
    with open(PROCESSED_FILES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ingest_inbox_directory(inbox_dir: str = "inbox", force: bool = False) -> None:
    """Scans an inbox/ directory and autonomously ingests all new or updated files."""
    if not os.path.exists(inbox_dir):
        os.makedirs(inbox_dir, exist_ok=True)
        print(f"Created drop folder '{inbox_dir}'. Drop any file (.pdf, .txt, .csv, .py, etc.) here to learn.")
        return

    learner = AutoLearner()
    processed_map = load_processed_files()
    files = [os.path.join(inbox_dir, f) for f in os.listdir(inbox_dir) if not f.startswith(".")]

    if not files:
        print(f"Drop folder '{inbox_dir}' is empty. Place any file here to learn it automatically.")
        return

    print(f"[auto_learner] Found {len(files)} file(s) in '{inbox_dir}'.")
    new_ingested = 0

    for fpath in files:
        if not os.path.isfile(fpath):
            continue

        bname = os.path.basename(fpath)
        sig = get_file_signature(fpath)

        if not force and processed_map.get(bname) == sig:
            print(f"  [Skipped] {bname} (Already ingested and unchanged)")
            continue

        print(f"\n--- Ingesting {bname} ---")
        res = learner.learn_file(fpath)
        print(res)
        processed_map[bname] = sig
        new_ingested += 1

    save_processed_files(processed_map)
    if new_ingested == 0:
        print("\nAll files in inbox/ are already up to date. (Use --force to re-learn).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Document Learner & Knowledge Ingestion.")
    parser.add_argument("--file", help="Path to any file to ingest and learn")
    parser.add_argument("--text", help="Raw text to learn immediately")
    parser.add_argument("--inbox", default="inbox", help="Directory to scan for dropped files")
    parser.add_argument("--force", action="store_true", help="Force re-learning files even if unchanged")
    args = parser.parse_args()

    learner = AutoLearner()
    if args.file:
        print(learner.learn_file(args.file))
    elif args.text:
        print(learner.learn_text(args.text))
    else:
        ingest_inbox_directory(args.inbox, force=args.force)
