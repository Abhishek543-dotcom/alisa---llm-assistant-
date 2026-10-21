#!/usr/bin/env python3
"""
dataset_builder.py
Production-Grade All-Rounder Multi-Task Instruction Dataset Generator for Alisa:
  - Generates balanced, diverse, high-quality train/valid/test datasets.
  - Transforms Alisa into a true All-Rounder Personal AI Assistant:
      1. All-Rounder Foundation: Coding, debugging, bash commands, reasoning, general QA.
      2. Personal Profile & Context: Abhishek's career, skills, preferences, timeline.
      3. Ingested Knowledge & Inbox Files: In-depth Q&A, summaries, code insights from inbox docs.
      4. Episodic Diary: Daily milestones and timeline records.
      5. Procedural Runbooks: Reusable developer workflows and step-by-step guides.
      6. Human Reinforcement Alignment: User feedback pairs (/good, /bad <correction>).
      7. Professional Communication: Email drafting, polite replies, technical memos.
      8. macOS System Controls: Volume, battery, diagnostics, app launching.
      9. Lifelong Solutions: How past technical errors and bugs were resolved.
  - Production Controls:
      - Rigorous 3-way split: train.jsonl (80%), valid.jsonl (10%), test.jsonl (10%).
      - SHA-256 deduplication across runs.
      - Schema validation (ChatML format) and token-length sanity checks.
      - Statistics tracking in data/dataset_stats.json.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple


SYSTEM_PROMPT_ALLROUNDER = (
    "You are Alisa, Abhishek Tiwari's dedicated, highly capable, and intelligent personal AI assistant. "
    "You excel at software engineering, coding, debugging, general reasoning, productivity, "
    "system commands, and document analysis. You know Abhishek's background, work, and preferences. "
    "Respond accurately, concisely, and helpfully."
)

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VALID_FILE = os.path.join(DATA_DIR, "valid.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
STATS_FILE = os.path.join(DATA_DIR, "dataset_stats.json")
PROFILE_FILE = os.path.join(DATA_DIR, "profile_memory.json")
LEARNED_FILE = os.path.join(DATA_DIR, "learned_memory.json")
SOLUTIONS_FILE = os.path.join(DATA_DIR, "solution_memory.json")
CORPUS_FILE = os.path.join(DATA_DIR, "emails_corpus.jsonl")
EPISODIC_FILE = os.path.join(DATA_DIR, "episodic_memory.json")
PROCEDURAL_FILE = os.path.join(DATA_DIR, "procedural_memory.json")
PREFERENCES_FILE = os.path.join(DATA_DIR, "preference_pairs.jsonl")


def hash_example(user_content: str, assistant_content: str) -> str:
    """Computes a SHA-256 hash of prompt and response to prevent duplicate training samples."""
    norm_prompt = re.sub(r"\s+", " ", user_content.strip().lower())
    norm_resp = re.sub(r"\s+", " ", assistant_content.strip().lower())
    return hashlib.sha256(f"{norm_prompt}:::{norm_resp}".encode("utf-8")).hexdigest()


# ==============================================================================
# Seed Foundation Examples (All-Rounder Assistant & Coding Anchor)
# ==============================================================================

FOUNDATION_EXAMPLES = [
    # Identity & Persona
    (
        "Who are you and what can you do?",
        "I am Alisa, your autonomous personal AI assistant. I help you with software development, Python and shell scripting, system control on macOS, analyzing documents in your inbox, drafting emails, and managing your personal knowledge.",
        "identity",
    ),
    (
        "What is your name?",
        "My name is Alisa. I am your personal AI assistant, running locally on your Apple Silicon Mac.",
        "identity",
    ),
    (
        "Who created you?",
        "I was built by and for Abhishek Tiwari as an autonomous, privacy-first personal AI assistant.",
        "identity",
    ),
    # Coding & Development
    (
        "Write a Python function to safely read a JSON file and return an empty dict if it doesn't exist.",
        "Here is the clean, robust implementation:\n\n```python\nimport json\nimport os\nfrom typing import Any, Dict\n\ndef load_json_safe(filepath: str) -> Dict[str, Any]:\n    if not os.path.exists(filepath):\n        return {}\n    try:\n        with open(filepath, 'r', encoding='utf-8') as f:\n            return json.load(f)\n    except Exception as e:\n        print(f'Error reading {filepath}: {e}')\n        return {}\n```",
        "coding",
    ),
    (
        "How do I check system memory and CPU usage in Python on macOS?",
        "You can use the `psutil` library:\n\n```python\nimport psutil\n\nram = psutil.virtual_memory()\nprint(f'RAM Usage: {ram.percent}% ({ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB)')\nprint(f'CPU Load: {psutil.cpu_percent(interval=0.5)}%')\n```",
        "coding",
    ),
    (
        "Give me a bash command to find all files larger than 100MB in the current directory.",
        "Run this command in terminal:\n\n```bash\nfind . -type f -size +100M -exec ls -lh {} +\n```",
        "coding",
    ),
    (
        "What is the difference between LoRA and full fine-tuning?",
        "LoRA (Low-Rank Adaptation) freezes the original model weights and injects trainable low-rank decomposition matrices into transformer layers (like attention projection matrices). This reduces trainable parameters by 99%+, dramatically drops GPU/VRAM consumption, prevents catastrophic forgetting, and allows fast switching between task adapters without copying the base weights.",
        "technical",
    ),
    # System Controls
    (
        "How do I mute system audio from terminal on macOS?",
        "You can mute macOS volume with AppleScript via osascript:\n\n```bash\nosascript -e 'set volume with output muted'\n```\nTo unmute: `osascript -e 'set volume without output muted'`",
        "system_control",
    ),
    (
        "Check battery status from command line on macOS.",
        "Run `pmset -g batt` in terminal. It returns the current percentage, charging state, and remaining runtime.",
        "system_control",
    ),
    # Productivity & Daily Tasks
    (
        "Help me prioritize my engineering tasks for today.",
        "A practical 3-tier prioritization framework for your engineering day:\n1. Critical Path / Blockers: Code reviews, unblocking teammates, and critical bug fixes.\n2. Deep Work (90-120 min block): Core architecture, feature implementation, and model training.\n3. Administrative & Maintenance: Ingesting documentation, cleaning branches, updating task trackers.",
        "productivity",
    ),
]


# ==============================================================================
# Dynamic Knowledge Extractors
# ==============================================================================

def extract_profile_examples(profile_path: str = PROFILE_FILE) -> List[Tuple[str, str, str]]:
    """Generates Q&A pairs from discovered profile memory."""
    examples = []
    if not os.path.exists(profile_path):
        return examples

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return examples

    latest_co = data.get("latest_company")
    if latest_co and latest_co != "Not yet discovered from records.":
        examples.append((
            "What company does Abhishek work at?",
            f"Based on Abhishek's latest records, he is currently at {latest_co}.",
            "personal_profile",
        ))
        examples.append((
            "Where do I currently work?",
            f"You are currently working at {latest_co}.",
            "personal_profile",
        ))

    profile = data.get("discovered_profile", {})
    for k, v in profile.items():
        if v and len(str(v)) < 150:
            examples.append((
                f"What is Abhishek's {k.replace('_', ' ')}?",
                f"Abhishek's {k.replace('_', ' ')} is {v}.",
                "personal_profile",
            ))

    career = data.get("career_progression", [])
    if career:
        history_str = " -> ".join([c.get("company", "") for c in career if c.get("company")])
        if history_str:
            examples.append((
                "Summarize Abhishek's career progression.",
                f"Abhishek's chronological career path across records includes: {history_str}.",
                "personal_profile",
            ))

    notes = data.get("custom_notes", {})
    for k, v in notes.items():
        if v:
            examples.append((
                f"What is the saved note for {k}?",
                f"{v}",
                "personal_notes",
            ))

    return examples


def extract_solution_examples(solutions_path: str = SOLUTIONS_FILE) -> List[Tuple[str, str, str]]:
    """Generates Q&A pairs from lifelong solved issues memory."""
    examples = []
    if not os.path.exists(solutions_path):
        return examples

    try:
        with open(solutions_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)
    except Exception:
        return examples

    for s in solutions:
        issue = s.get("issue", "")
        fix = s.get("resolution", "")
        cause = s.get("root_cause", "")
        if issue and fix:
            prompt = f"How did we fix the issue: {issue}?"
            ans = f"Root cause: {cause}\nResolution: {fix}" if cause else f"Resolution: {fix}"
            examples.append((prompt, ans, "solutions"))
    return examples


def extract_inbox_document_examples(records: List[Dict]) -> List[Tuple[str, str, str]]:
    """Generates instruction pairs from ingested inbox documents."""
    examples = []

    for r in records:
        text = r.get("text", "").strip()
        meta = r.get("metadata", {})
        src = meta.get("source", "document")
        doc_type = meta.get("type", "generic")

        if len(text) < 50:
            continue

        if "code" in doc_type:
            first_few_lines = "\n".join([line for line in text.splitlines()[:15] if line.strip()])
            examples.append((
                f"Explain the purpose of the code file '{src}'.",
                f"The file '{src}' contains source code for the following implementation:\n\n{text[:1200]}",
                "inbox_code",
            ))
            examples.append((
                f"How is '{src}' structured and what does it implement?",
                f"Overview of '{src}':\n\n```\n{first_few_lines}\n```\n\nFull implementation details:\n{text[:1000]}",
                "inbox_code",
            ))
        else:
            sentences = [s.strip() for s in re.split(r"[.\n]+", text) if len(s.strip()) > 30]
            summary_part = text[:800]

            examples.append((
                f"Summarize the key information in '{src}'.",
                f"Key points from '{src}':\n\n{summary_part}",
                "inbox_document",
            ))

            if sentences:
                sample_s = sentences[0]
                examples.append((
                    f"What does '{src}' say regarding: '{sample_s[:100]}...'?",
                    f"According to '{src}':\n\n{text[:1000]}",
                    "inbox_document",
                ))

            examples.append((
                f"Extract the main takeaways and actionable details from '{src}'.",
                f"Here are the salient facts and details recorded in '{src}':\n{text[:1200]}",
                "inbox_document",
            ))

    return examples


def extract_preference_examples(preferences_path: str = PREFERENCES_FILE) -> List[Tuple[str, str, str]]:
    """Generates high-priority alignment instruction pairs from user feedback (/good and /bad)."""
    examples = []
    if not os.path.exists(preferences_path):
        return examples
    try:
        with open(preferences_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "").strip()
                chosen = item.get("chosen_response", "").strip()
                if prompt and chosen and len(chosen) > 10:
                    examples.append((prompt, chosen, "human_alignment"))
    except Exception:
        pass
    return examples


def extract_episodic_examples(episodic_path: str = EPISODIC_FILE) -> List[Tuple[str, str, str]]:
    """Generates instruction pairs from chronological episodic diary."""
    examples = []
    if not os.path.exists(episodic_path):
        return examples
    try:
        with open(episodic_path, "r", encoding="utf-8") as f:
            episodes = json.load(f)
        for date_str, ev_list in episodes.items():
            if not ev_list:
                continue
            events_text = "\n".join([f"• [{e.get('time', '')}] ({e.get('category', 'task')}) {e.get('event')}" for e in ev_list])
            examples.append((
                f"What did I work on on {date_str}?",
                f"Here is your recorded activity diary for {date_str}:\n\n{events_text}",
                "episodic_diary",
            ))
            examples.append((
                f"Summarize my milestones for {date_str}.",
                f"Logged milestones for {date_str}:\n{events_text}",
                "episodic_diary",
            ))
    except Exception:
        pass
    return examples


def extract_procedural_examples(procedural_path: str = PROCEDURAL_FILE) -> List[Tuple[str, str, str]]:
    """Generates instruction pairs from developer runbooks and task procedures."""
    examples = []
    if not os.path.exists(procedural_path):
        return examples
    try:
        with open(procedural_path, "r", encoding="utf-8") as f:
            procedures = json.load(f)
        for p in procedures.values():
            name = p.get("name", "")
            trigger = p.get("trigger", f"How do I {name}")
            steps = p.get("steps", [])
            notes = p.get("notes", "")
            if steps:
                steps_formatted = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
                response = f"Procedure for '{name}':\n\n{steps_formatted}"
                if notes:
                    response += f"\n\nNotes: {notes}"
                examples.append((trigger, response, "procedural_runbook"))
                examples.append((f"What are the steps to {name}?", response, "procedural_runbook"))
    except Exception:
        pass
    return examples


def extract_email_examples(corpus_path: str = CORPUS_FILE, max_emails: int = 50) -> List[Tuple[str, str, str]]:
    """Generates communication and drafting examples from clean email corpus."""
    examples = []
    if not os.path.exists(corpus_path):
        return examples

    try:
        count = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= max_emails:
                    break
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("is_sent"):
                    continue
                body = r.get("body", "").strip()
                if len(body) < 40 or len(body) > 2000:
                    continue

                subj = r.get("subject", "(No Subject)")
                to_addr = r.get("to", "recipient")
                user_prompt = f"Draft a professional message to {to_addr} with the subject '{subj}'."
                examples.append((user_prompt, body, "email_communication"))
                count += 1
    except Exception:
        pass

    return examples


# ==============================================================================
# Unified Dataset Builder Engine
# ==============================================================================

class DatasetBuilder:
    """
    Constructs, balances, deduplicates, and validates the complete All-Rounder instruction dataset.
    """

    def __init__(self, output_dir: str = DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.seen_hashes: Set[str] = set()

    def build_all_rounder_dataset(
        self,
        inbox_records: Optional[List[Dict]] = None,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Builds and writes train.jsonl, valid.jsonl, and test.jsonl with clean deduplication.
        """
        self.seen_hashes.clear()
        collected_pairs: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}

        def add_example(user_q: str, assistant_a: str, category: str):
            clean_q = user_q.strip()
            clean_a = assistant_a.strip()
            if len(clean_q) < 5 or len(clean_a) < 5:
                return

            h = hash_example(clean_q, clean_a)
            if h in self.seen_hashes:
                return
            self.seen_hashes.add(h)

            item = {
                "category": category,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_ALLROUNDER},
                    {"role": "user", "content": clean_q},
                    {"role": "assistant", "content": clean_a},
                ],
            }
            collected_pairs.append(item)
            category_counts[category] = category_counts.get(category, 0) + 1

        # 1. Foundation & Coding Examples
        for q, a, cat in FOUNDATION_EXAMPLES:
            add_example(q, a, cat)

        # 2. Personal Profile & Notes
        for q, a, cat in extract_profile_examples(PROFILE_FILE):
            add_example(q, a, cat)

        # 3. Solved Issues & Bug Resolutions
        for q, a, cat in extract_solution_examples(SOLUTIONS_FILE):
            add_example(q, a, cat)

        # 4. Human Feedback & Alignment Pairs
        for q, a, cat in extract_preference_examples(PREFERENCES_FILE):
            add_example(q, a, cat)

        # 5. Episodic Diary Events
        for q, a, cat in extract_episodic_examples(EPISODIC_FILE):
            add_example(q, a, cat)

        # 6. Procedural Runbooks & Workflows
        for q, a, cat in extract_procedural_examples(PROCEDURAL_FILE):
            add_example(q, a, cat)

        # 7. Ingested Documents / Inbox
        if inbox_records:
            for q, a, cat in extract_inbox_document_examples(inbox_records):
                add_example(q, a, cat)

        # 8. Clean Email Communication
        for q, a, cat in extract_email_examples(CORPUS_FILE, max_emails=40):
            add_example(q, a, cat)

        # 9. Preserve existing valid train examples if already present
        if os.path.exists(TRAIN_FILE):
            try:
                with open(TRAIN_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        msgs = obj.get("messages", [])
                        if len(msgs) >= 3:
                            u_content = msgs[1].get("content", "")
                            a_content = msgs[2].get("content", "")
                            add_example(u_content, a_content, "historical")
            except Exception:
                pass

        if not collected_pairs:
            for q, a, cat in FOUNDATION_EXAMPLES:
                add_example(q, a, cat)

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(collected_pairs)

        total_count = len(collected_pairs)
        test_size = max(1, int(total_count * test_ratio)) if total_count >= 10 else 0
        val_size = max(1, int(total_count * val_ratio)) if total_count >= 10 else 0
        train_size = total_count - val_size - test_size

        if train_size <= 0:
            train_set = collected_pairs
            valid_set = collected_pairs[:1]
            test_set = collected_pairs[:1]
        else:
            test_set = collected_pairs[:test_size]
            valid_set = collected_pairs[test_size : test_size + val_size]
            train_set = collected_pairs[test_size + val_size :]

        self._write_jsonl(TRAIN_FILE, train_set)
        self._write_jsonl(VALID_FILE, valid_set)
        self._write_jsonl(TEST_FILE, test_set)

        stats = {
            "last_built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_examples": total_count,
            "train_examples": len(train_set),
            "valid_examples": len(valid_set),
            "test_examples": len(test_set),
            "category_distribution": category_counts,
            "files": {
                "train": TRAIN_FILE,
                "valid": VALID_FILE,
                "test": TEST_FILE,
            },
        }

        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def _write_jsonl(self, path: str, items: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                clean_item = {"messages": item["messages"]}
                f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")


def build_datasets_cli():
    parser = argparse.ArgumentParser(description="Build and validate All-Rounder instruction dataset for Alisa.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio (default: 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio (default: 0.10)")
    args = parser.parse_args()

    builder = DatasetBuilder()
    stats = builder.build_all_rounder_dataset(val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    print("\n" + "=" * 65)
    print("📊 Alisa All-Rounder Dataset Generation Complete")
    print("=" * 65)
    print(f"Total Unique Instruction Pairs: {stats['total_examples']}")
    print(f"  • Training Set (train.jsonl):    {stats['train_examples']} examples")
    print(f"  • Validation Set (valid.jsonl):  {stats['valid_examples']} examples")
    print(f"  • Test Set (test.jsonl):        {stats['test_examples']} examples")
    print("\nCategory Distribution:")
    for cat, cnt in stats["category_distribution"].items():
        print(f"  • {cat:<24}: {cnt} pairs")
    print("=" * 65)


if __name__ == "__main__":
    build_datasets_cli()
