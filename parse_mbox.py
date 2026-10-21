#!/usr/bin/env python3
"""
parse_mbox.py
Parses a Gmail Takeout .mbox file, filters out Spam, Trash, and automated promotions/newsletters,
and generates:
  1. data/emails_corpus.jsonl  -> Clean personal/direct email corpus for local RAG indexing
  2. data/train.jsonl          -> Instruction fine-tuning dataset (90% split)
  3. data/valid.jsonl          -> Validation dataset (10% split)
"""

import argparse
import collections
import email
import email.header
import email.utils
import html.parser
import json
import mailbox
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union


SKIP_LABELS = {
    "spam",
    "trash",
    "junk",
    "category promotions",
    "category social",
    "category updates",
    "category forums",
    "muted",
}

AUTOMATED_SENDER_PATTERNS = re.compile(
    r"(no-?reply|donotreply|do-not-reply|mailer-daemon|postmaster|notifications?|"
    r"alerts?|newsletter|digest|updates?@|bounce|automated|support@|info@|marketing@)",
    re.IGNORECASE,
)


class ReadOnlyMbox(mailbox.mbox):
    """Safe read-only mbox wrapper that handles OS permissions and read-only files gracefully."""

    def __init__(self, path: str, factory=None, create: bool = False):
        try:
            super().__init__(path, factory=factory, create=create)
        except (PermissionError, OSError):
            mailbox.Mailbox.__init__(self, path, factory, create)
            self._file = open(self._path, "rb")
            self._toc = None
            self._next_key = 0
            self._pending = False
            self._pending_sync = False
            self._locked = False
            self._file_length = None


class HTMLTextExtractor(html.parser.HTMLParser):
    """Lightweight HTML-to-text converter that strips scripts and styles."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "head", "noscript"):
            self._ignore_depth += 1
        elif tag_lower in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "head", "noscript") and self._ignore_depth > 0:
            self._ignore_depth -= 1
        elif tag_lower in ("p", "div", "tr", "li"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def safe_decode_payload(payload: Union[bytes, str, None], charset: Optional[str] = None) -> str:
    """Safely decodes a byte payload, gracefully handling non-standard or invalid encodings like 'binary'."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, (bytes, bytearray)):
        return str(payload)

    if charset:
        clean_charset = str(charset).strip().strip("'\"").lower()
        if ";" in clean_charset:
            clean_charset = clean_charset.split(";")[0].strip()
        # Normalise common non-standard or invalid charset aliases found in email headers
        aliases = {
            "binary": "utf-8",
            "7bit": "ascii",
            "8bit": "utf-8",
            "utf8": "utf-8",
            "cp-1252": "cp1252",
            "windows-1252": "cp1252",
            "iso-8859-8-i": "iso-8859-8",
            "unicode-1-1-utf-8": "utf-8",
            "default": "utf-8",
            "none": "utf-8",
            "unknown": "utf-8",
            "x-unknown": "utf-8",
        }
        clean_charset = aliases.get(clean_charset, clean_charset)
        try:
            return payload.decode(clean_charset, errors="replace")
        except (LookupError, ValueError, TypeError):
            pass

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(enc, errors="replace")
        except Exception:
            continue

    return payload.decode("latin-1", errors="replace")


def decode_mime_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        parts = email.header.decode_header(value)
        decoded = []
        for content, charset in parts:
            if isinstance(content, bytes):
                decoded.append(safe_decode_payload(content, charset))
            else:
                decoded.append(str(content))
        return "".join(decoded).strip()
    except Exception:
        return str(value).strip()


def html_to_text(html_content: str) -> str:
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html_content).strip()


def extract_body(msg: email.message.Message) -> str:
    plain_parts: List[str] = []
    html_parts: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            # Only process textual mime parts; skip attachments and media
            if content_type not in ("text/plain", "text/html"):
                continue
            disposition = str(part.get("Content-Disposition") or "").lower()
            if "attachment" in disposition:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            text = safe_decode_payload(payload, part.get_content_charset())
            if content_type == "text/plain":
                plain_parts.append(text)
            elif content_type == "text/html":
                html_parts.append(html_to_text(text))
    else:
        content_type = msg.get_content_type()
        if content_type in ("text/plain", "text/html"):
            payload = msg.get_payload(decode=True)
            if payload:
                text = safe_decode_payload(payload, msg.get_content_charset())
                if content_type == "text/html":
                    html_parts.append(html_to_text(text))
                else:
                    plain_parts.append(text)

    combined = "\n".join(plain_parts).strip() if plain_parts else "\n".join(html_parts).strip()
    # Normalize excessive whitespace
    combined = re.sub(r"\r\n", "\n", combined)
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined.strip()


def strip_quoted_reply(text: str) -> str:
    """Removes quoted reply blocks ('On ... wrote:', '>' lines, Outlook headers) for clean training targets."""
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        if re.match(r"^On .+ wrote:$", line.strip()):
            break
        if re.match(r"^-{2,}\s*Original Message\s*-{2,}", line.strip(), re.IGNORECASE):
            break
        if re.match(r"^From:\s+.+", line.strip()) and len(cleaned_lines) > 2:
            break
        if line.strip().startswith(">"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def should_skip_message(msg: email.message.Message, is_sent_by_user: bool) -> bool:
    labels_raw = decode_mime_header(msg.get("X-Gmail-Labels", ""))
    labels = {lbl.strip().lower() for lbl in labels_raw.split(",") if lbl.strip()}

    # Always filter out Spam, Trash, and promotional categories
    if labels & SKIP_LABELS:
        return True

    # Keep messages explicitly sent by the user
    if is_sent_by_user or "sent" in labels:
        return False

    # Filter out automated bulk/newsletter messages
    from_header = decode_mime_header(msg.get("From", ""))
    if AUTOMATED_SENDER_PATTERNS.search(from_header):
        return True
    if msg.get("List-Unsubscribe") or msg.get("List-Id"):
        return True
    precedence = (msg.get("Precedence") or "").lower()
    if precedence in ("bulk", "list", "junk"):
        return True
    auto_submitted = (msg.get("Auto-Submitted") or "").lower()
    if auto_submitted and auto_submitted != "no":
        return True

    return False


def detect_user_email(mbox: mailbox.mbox, sample_limit: int = 5000) -> Optional[str]:
    sent_senders: collections.Counter = collections.Counter()
    for idx, msg in enumerate(mbox):
        if idx >= sample_limit:
            break
        try:
            labels_raw = decode_mime_header(msg.get("X-Gmail-Labels", "")).lower()
            if "sent" in labels_raw:
                _, addr = email.utils.parseaddr(decode_mime_header(msg.get("From", "")))
                if addr:
                    sent_senders[addr.lower()] += 1
        except Exception:
            continue
    if sent_senders:
        return sent_senders.most_common(1)[0][0]
    return None


def find_default_mbox() -> str:
    """Finds default mbox in inbox/, current dir, or Downloads."""
    candidates = [
        "inbox/All mail Including Spam and Trash-002.mbox",
        "./All mail Including Spam and Trash-002.mbox",
        os.path.expanduser("~/Downloads/All mail Including Spam and Trash-002.mbox"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Check any .mbox in inbox/ or current dir
    if os.path.exists("inbox"):
        for f in os.listdir("inbox"):
            if f.endswith(".mbox"):
                return os.path.join("inbox", f)
    for f in os.listdir("."):
        if f.endswith(".mbox"):
            return f
    return os.path.expanduser("~/Downloads/All mail Including Spam and Trash-002.mbox")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and filter Gmail mbox for RAG and LoRA fine-tuning.")
    parser.add_argument(
        "--mbox",
        default=find_default_mbox(),
        help="Path to the Gmail Takeout .mbox file",
    )
    parser.add_argument("--output-dir", default="data", help="Output directory for JSONL datasets")
    parser.add_argument("--user-email", default=None, help="Your primary Gmail address (auto-detected if omitted)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--max-emails", type=int, default=None, help="Optional limit on number of emails to parse (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.mbox):
        raise FileNotFoundError(f"Mbox file not found at: {args.mbox}")

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        mbox = ReadOnlyMbox(args.mbox, create=False)
    except PermissionError:
        raise PermissionError(
            f"\n\n[Permission Denied by macOS TCC]\n"
            f"macOS restricted your terminal app from accessing files in Downloads: {args.mbox}\n\n"
            f"To fix this immediately, move the mbox file into this project directory (instant, 0 extra disk space):\n"
            f"  mv \"{args.mbox}\" ./\n"
            f"Then run:\n"
            f"  ./run_pipeline.sh\n"
        ) from None

    user_email = (args.user_email or detect_user_email(mbox) or "").lower()
    print(f"[parse_mbox] Detected user email: {user_email or 'Unknown (relying on Sent label)'}")

    corpus_path = os.path.join(args.output_dir, "emails_corpus.jsonl")
    existing_message_ids = set()
    corpus_records: List[Dict] = []
    messages_by_id: Dict[str, Dict] = {}

    if os.path.exists(corpus_path):
        print(f"[parse_mbox] Found existing corpus at {corpus_path}. Reading existing message IDs for incremental merge...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                corpus_records.append(r)
                mid = r.get("message_id")
                if mid:
                    existing_message_ids.add(mid)
                    messages_by_id[mid] = r

        print(f"[parse_mbox] Loaded {len(corpus_records)} existing emails. Scanning new .mbox for newly added emails...")

    skipped_count = 0
    new_added_count = 0

    for idx, msg in enumerate(mbox):
        if args.max_emails is not None and idx >= args.max_emails:
            break
        if (idx + 1) % 5000 == 0:
            print(f"[parse_mbox] Processed {idx + 1} emails ({new_added_count} new kept, {skipped_count} skipped)...")

        try:
            msg_id = (msg.get("Message-ID") or f"<msg-{idx}>").strip()
            if msg_id in existing_message_ids:
                skipped_count += 1
                continue

            from_raw = decode_mime_header(msg.get("From", ""))
            _, from_addr = email.utils.parseaddr(from_raw)
            from_addr_lower = from_addr.lower()

            labels_raw = decode_mime_header(msg.get("X-Gmail-Labels", ""))
            labels_list = [lbl.strip() for lbl in labels_raw.split(",") if lbl.strip()]
            is_sent = ("sent" in {l.lower() for l in labels_list}) or (
                bool(user_email) and from_addr_lower == user_email
            )

            if should_skip_message(msg, is_sent_by_user=is_sent):
                skipped_count += 1
                continue

            body = extract_body(msg)
            if len(body) < 20:
                skipped_count += 1
                continue

            in_reply_to = (msg.get("In-Reply-To") or "").strip()
            subject = decode_mime_header(msg.get("Subject", "(No Subject)"))
            to_raw = decode_mime_header(msg.get("To", ""))
            date_raw = decode_mime_header(msg.get("Date", ""))

            record = {
                "id": f"email_{len(corpus_records)}",
                "message_id": msg_id,
                "in_reply_to": in_reply_to,
                "date": date_raw,
                "from": from_raw,
                "to": to_raw,
                "subject": subject,
                "labels": labels_list,
                "is_sent": is_sent,
                "body": body,
            }
            corpus_records.append(record)
            messages_by_id[msg_id] = record
            existing_message_ids.add(msg_id)
            new_added_count += 1
        except Exception:
            skipped_count += 1
            continue

    print(f"[parse_mbox] Added {new_added_count} new emails (Total corpus: {len(corpus_records)} emails). Writing to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build instruction-tuning examples from user's sent messages and replies
    sft_examples: List[Dict] = []
    system_prompt = (
        "You are Abhishek Tiwari's personal AI assistant trained on his personal email history. "
        "Respond accurately using his personal context, communication style, and tone."
    )

    for rec in corpus_records:
        if not rec["is_sent"]:
            continue
        clean_reply = strip_quoted_reply(rec["body"])
        if len(clean_reply) < 25:
            continue

        parent = messages_by_id.get(rec["in_reply_to"])
        if parent:
            parent_body = strip_quoted_reply(parent["body"]) or parent["body"][:1500]
            user_prompt = (
                f"Reply to the following email from {parent['from']} "
                f"(Subject: {rec['subject']}, Date: {parent['date']}):\n\n{parent_body[:2000]}"
            )
        else:
            user_prompt = (
                f"Write an email to {rec['to']} with the subject '{rec['subject']}' "
                f"(Date context: {rec['date']})."
            )

        sft_examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": clean_reply[:3000]},
                ]
            }
        )

    random.seed(42)
    random.shuffle(sft_examples)
    val_size = max(1, int(len(sft_examples) * args.val_ratio)) if len(sft_examples) > 1 else 0
    valid_set = sft_examples[:val_size]
    train_set = sft_examples[val_size:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    valid_path = os.path.join(args.output_dir, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    try:
        from memory_manager import MemoryManager
        mem = MemoryManager()
        mem.sync_from_corpus(corpus_path)
        print(f"[parse_mbox] Successfully synchronized memory profile. Latest company detected: '{mem.get_latest_company()}'")
    except Exception as e:
        print(f"[parse_mbox] Note: memory profile sync skipped ({e})")

    # Build balanced All-Rounder instruction dataset with 3-way split (train/valid/test)
    try:
        from dataset_builder import DatasetBuilder
        builder = DatasetBuilder(output_dir=args.output_dir)
        stats = builder.build_all_rounder_dataset(val_ratio=args.val_ratio, test_ratio=args.val_ratio)
        print(f"[parse_mbox] Generated balanced All-Rounder dataset: {stats['train_examples']} train, {stats['valid_examples']} valid, {stats['test_examples']} test examples.")
    except Exception as e:
        print(f"[parse_mbox] Note: DatasetBuilder warning ({e})")

    print(f"[parse_mbox] Kept {len(corpus_records)} clean emails (filtered {skipped_count} spam/trash/promo emails).")
    print(f"[parse_mbox] Saved RAG corpus to {corpus_path}")


if __name__ == "__main__":
    main()
