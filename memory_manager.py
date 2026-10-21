#!/usr/bin/env python3
"""
memory_manager.py
Zero-Hardcoding Dynamic Memory & Profile System:
  - ZERO static assumptions or hardcoded company names.
  - Everything is learned dynamically from emails, documents, and user interactions.
  - Chronological awareness: automatically determines the latest company and role based on the newest timestamp.
  - Persists dynamic knowledge in data/profile_memory.json.
"""

import email.utils
import json
import os
import re
from typing import Any, Dict, List, Optional


MEMORY_FILE = "data/profile_memory.json"

DISALLOWED_SUBSTRINGS = [
    "experience", "designation", "location", "salary", "ctc", "notice",
    "qualification", "status", "payroll", "offer", "preferred",
    "contract", "working", "lwd", "date", "thanks", "regards",
    "client", "job", "good day", "opportunity", ":", "serving", "http"
]


def clean_entity_value(raw: str) -> Optional[str]:
    """Cleans an extracted entity value dynamically without hardcoding."""
    cleaned = re.sub(r"[\*\_#`]", "", raw).strip()
    # Strip leading/trailing punctuation/bullets
    cleaned = re.sub(r"^[\s\:\-\>\•\(\)]+", "", cleaned).strip()
    cleaned = re.sub(r"[\s\:\-]+$", "", cleaned).strip()

    if not cleaned or len(cleaned) < 2 or len(cleaned) > 50:
        return None

    lower = cleaned.lower()
    if any(sub in lower for sub in DISALLOWED_SUBSTRINGS):
        return None

    return cleaned


class MemoryManager:
    def __init__(self, memory_path: str = MEMORY_FILE):
        self.memory_path = memory_path
        self.data: Dict[str, Any] = self._load()

    def _default_memory(self) -> Dict[str, Any]:
        """Empty baseline structure - ZERO hardcoded facts."""
        return {
            "discovered_profile": {},
            "career_progression": [],
            "custom_notes": {},
            "latest_company": None,
            "last_synced": None,
        }

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        default_data = self._default_memory()
        self._save(default_data)
        return default_data

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is None:
            data = self.data
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def remember(self, key: str, value: str) -> str:
        """Stores or updates a custom user fact dynamically."""
        key_clean = key.strip().lower().replace(" ", "_")
        self.data["custom_notes"][key_clean] = value.strip()
        self._save()
        return f"Stored in memory: '{key_clean}' = '{value.strip()}'"

    def recall(self, key: Optional[str] = None) -> Any:
        if key:
            key_clean = key.strip().lower().replace(" ", "_")
            if key_clean in self.data.get("discovered_profile", {}):
                return self.data["discovered_profile"][key_clean]
            if key_clean in self.data.get("custom_notes", {}):
                return self.data["custom_notes"][key_clean]
            return self.data.get(key_clean)
        return self.data

    def get_latest_company(self) -> str:
        if self.data.get("latest_company"):
            return self.data["latest_company"]
        timeline = self.data.get("career_progression", [])
        if timeline:
            return timeline[-1]["company"]
        return "Not yet discovered from records."

    def get_summary(self) -> str:
        """Generates dynamic memory summary for LLM prompt without any hardcoding."""
        lines = []

        # Discovered profile attributes
        profile = self.data.get("discovered_profile", {})
        for k, v in profile.items():
            readable_k = k.replace("_", " ").title()
            if isinstance(v, list):
                lines.append(f"{readable_k}: {', '.join(str(x) for x in v)}")
            else:
                lines.append(f"{readable_k}: {v}")

        if self.data.get("latest_company"):
            lines.append(f"Current / Latest Company: {self.data['latest_company']}")

        # Chronological career progression
        timeline = self.data.get("career_progression", [])
        if timeline:
            lines.append("Chronological Career Progression (Discovered from emails):")
            for item in timeline:
                extra = []
                if item.get("ctc"):
                    extra.append(f"CTC: {item['ctc']}")
                if item.get("experience"):
                    extra.append(f"Exp: {item['experience']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                lines.append(f"  • {item['date']}: {item['company']}{extra_str}")

        # Custom notes
        notes = self.data.get("custom_notes", {})
        if notes:
            lines.append("User Notes & Preferences:")
            for k, v in notes.items():
                lines.append(f"  • {k}: {v}")

        if not lines:
            return "No profile records discovered yet. Ingest emails or documents to learn."

        return "\n".join(lines)

    def sync_from_corpus(self, corpus_path: str = "data/emails_corpus.jsonl") -> Dict[str, Any]:
        """
        Scans email corpus dynamically to extract company transitions, CTC, education,
        and personal details WITHOUT any hardcoded names.
        """
        if not os.path.exists(corpus_path):
            return self.data

        # Dynamic regex extractors for common resume/intro fields
        pattern_company = re.compile(
            r"(?:Current Company|Current Organization|Current Employer)\s*[:\-]\s*([^\n\r,]+)",
            re.IGNORECASE,
        )
        pattern_exp = re.compile(
            r"(?:Total(?: IT)? Experience)\s*[:\-]\s*([^\n\r]+)",
            re.IGNORECASE,
        )
        pattern_ctc = re.compile(
            r"(?:Current CTC|CTC \(Fixed \+ Variable\)|Fixed CTC)\s*[:\-]\s*([^\n\r]+)",
            re.IGNORECASE,
        )
        pattern_loc = re.compile(
            r"(?:Current Location)\s*[:\-]\s*([^\n\r]+)",
            re.IGNORECASE,
        )
        pattern_father = re.compile(
            r"(?:Father(?:'s)? Name)\s*[:\-]\s*([^\n\r]+)",
            re.IGNORECASE,
        )

        detected_events = []
        discovered_info = dict(self.data.get("discovered_profile", {}))

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                body = rec.get("body", "")

                try:
                    dt = email.utils.parsedate_to_datetime(rec.get("date", ""))
                    ts = dt.timestamp()
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    continue

                # Check father name
                m_father = pattern_father.search(body)
                if m_father:
                    val = clean_entity_value(m_father.group(1))
                    if val and "father_name" not in discovered_info:
                        discovered_info["father_name"] = val

                # Check location
                m_loc = pattern_loc.search(body)
                if m_loc:
                    val = clean_entity_value(m_loc.group(1))
                    if val and "location" not in discovered_info:
                        discovered_info["location"] = val

                # Check company
                m_comp = pattern_company.search(body)
                if m_comp:
                    comp_name = clean_entity_value(m_comp.group(1))
                    if comp_name:
                        m_exp = pattern_exp.search(body)
                        m_ctc = pattern_ctc.search(body)
                        detected_events.append({
                            "timestamp": ts,
                            "date": date_str,
                            "company": comp_name,
                            "experience": clean_entity_value(m_exp.group(1)) if m_exp else None,
                            "ctc": clean_entity_value(m_ctc.group(1)) if m_ctc else None,
                        })

        if detected_events:
            # Sort chronologically by timestamp (newest last)
            detected_events.sort(key=lambda x: x["timestamp"])

            # Clean and deduplicate chronological timeline
            timeline = []
            seen_companies = set()
            for ev in detected_events:
                c_key = ev["company"].lower()
                if c_key not in seen_companies:
                    seen_companies.add(c_key)
                    timeline.append({
                        "date": ev["date"],
                        "company": ev["company"],
                        "ctc": ev.get("ctc"),
                        "experience": ev.get("experience"),
                    })

            self.data["career_progression"] = timeline
            self.data["latest_company"] = detected_events[-1]["company"]
            if detected_events[-1].get("ctc"):
                discovered_info["latest_ctc"] = detected_events[-1]["ctc"]
            if detected_events[-1].get("experience"):
                discovered_info["total_experience"] = detected_events[-1]["experience"]

        self.data["discovered_profile"] = discovered_info
        self.data["last_synced"] = email.utils.formatdate(usegmt=True)
        self._save()
        return self.data


if __name__ == "__main__":
    mgr = MemoryManager()
    mgr.sync_from_corpus()
    print("=== DYNAMICALLY DISCOVERED MEMORY (0 HARDCODING) ===")
    print(mgr.get_summary())
