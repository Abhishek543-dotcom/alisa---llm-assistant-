#!/usr/bin/env python3
"""
memory_manager.py
Multi-Tier Cognitive Memory Engine for Alisa:
  1. Semantic & Profile Memory (data/profile_memory.json):
     - Discovered profile, career progression, skills, CTC, and custom preferences.
  2. Episodic Memory (data/episodic_memory.json):
     - Chronological daily diary of activities, milestones, and sessions.
  3. Procedural Memory (data/procedural_memory.json):
     - Reusable developer workflows, bash runbooks, and task execution steps.
  4. Human Feedback & Preference Memory (data/preference_pairs.jsonl):
     - Reinforcement learning via user feedback (/good, /bad <correction>).
  5. Lifelong Error & Bug Resolutions (data/solution_memory.json):
     - Troubleshot issues, root causes, and verified fixes.
  6. Interaction History (data/conversation_history.jsonl):
     - Lifelong multi-turn conversation memory.
"""

import datetime
import email.utils
import json
import os
import re
import time
from typing import Any, Dict, List, Optional


MEMORY_FILE = "data/profile_memory.json"
HISTORY_FILE = "data/conversation_history.jsonl"
SOLUTIONS_FILE = "data/solution_memory.json"
EPISODIC_FILE = "data/episodic_memory.json"
PROCEDURAL_FILE = "data/procedural_memory.json"
PREFERENCES_FILE = "data/preference_pairs.jsonl"

DISALLOWED_SUBSTRINGS = [
    "experience", "designation", "location", "salary", "ctc", "notice",
    "qualification", "status", "payroll", "offer", "preferred",
    "contract", "working", "lwd", "date", "thanks", "regards",
    "client", "job", "good day", "opportunity", ":", "serving", "http"
]


def clean_entity_value(raw: str) -> Optional[str]:
    """Cleans an extracted entity value dynamically without hardcoding."""
    cleaned = re.sub(r"[\*\_#`]", "", raw).strip()
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

    # =========================================================================
    # 1. Interaction History
    # =========================================================================

    def log_interaction(self, query: str, answer: str, source: str = "console") -> None:
        """Appends conversation turn to persistent history for lifelong memory."""
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source,
            "query": query.strip(),
            "answer": answer.strip(),
        }
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent_history(self, limit: int = 5) -> List[Dict[str, str]]:
        """Retrieves the last N conversation interactions."""
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return [json.loads(l) for l in lines[-limit:]]
        except Exception:
            return []

    # =========================================================================
    # 2. Episodic Daily Memory (Daily diary & timeline)
    # =========================================================================

    def log_episode(self, event: str, category: str = "general", details: str = "") -> None:
        """Logs an episodic event under today's date bucket."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M")
        episodes = self.load_episodes()

        if today not in episodes:
            episodes[today] = []

        episodes[today].append({
            "time": now_time,
            "category": category,
            "event": event.strip(),
            "details": details.strip(),
        })

        os.makedirs(os.path.dirname(EPISODIC_FILE) or ".", exist_ok=True)
        with open(EPISODIC_FILE, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2, ensure_ascii=False)

    def load_episodes(self) -> Dict[str, List[Dict[str, str]]]:
        if not os.path.exists(EPISODIC_FILE):
            return {}
        try:
            with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_recent_episodes_summary(self, days: int = 3) -> str:
        """Returns readable episodic recap of recent days."""
        episodes = self.load_episodes()
        if not episodes:
            return "No episodic milestones logged yet."

        sorted_dates = sorted(episodes.keys(), reverse=True)[:days]
        lines = []
        for d in sorted_dates:
            lines.append(f"📅 [{d}]:")
            for ev in episodes[d]:
                det = f" ({ev['details']})" if ev.get("details") else ""
                lines.append(f"  • {ev.get('time', '')} [{ev.get('category', 'task')}] {ev.get('event')}{det}")
        return "\n".join(lines)

    # =========================================================================
    # 3. Procedural Memory (Workflows, Runbooks, Step-by-Step guides)
    # =========================================================================

    def save_procedure(self, name: str, steps: List[str], trigger: str = "", notes: str = "") -> str:
        """Saves a developer runbook or task workflow."""
        procedures = self.load_procedures()
        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower().strip())
        procedures[slug] = {
            "name": name.strip(),
            "trigger": trigger.strip() or f"How to {name}",
            "steps": steps,
            "notes": notes.strip(),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        os.makedirs(os.path.dirname(PROCEDURAL_FILE) or ".", exist_ok=True)
        with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
            json.dump(procedures, f, indent=2, ensure_ascii=False)
        return f"Saved procedure: '{name}' with {len(steps)} steps."

    def load_procedures(self) -> Dict[str, Any]:
        if not os.path.exists(PROCEDURAL_FILE):
            return {}
        try:
            with open(PROCEDURAL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def search_procedures(self, query: str) -> List[Dict[str, Any]]:
        procedures = self.load_procedures()
        q_words = query.lower().split()
        matches = []
        for p in procedures.values():
            haystack = f"{p['name']} {p['trigger']} {' '.join(p['steps'])} {p.get('notes', '')}".lower()
            if any(w in haystack for w in q_words):
                matches.append(p)
        return matches

    # =========================================================================
    # 4. Human Feedback & Preference Memory (RLHF / DPO Pairs)
    # =========================================================================

    def record_feedback(
        self,
        prompt: str,
        response: str,
        rating: str,
        correction: str = "",
    ) -> str:
        """
        Records human reinforcement feedback (/good or /bad <correction>).
        Feeds into preference alignment and continuous fine-tuning.
        """
        os.makedirs(os.path.dirname(PREFERENCES_FILE) or ".", exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt.strip(),
            "rating": rating.lower().strip(),  # "positive" or "negative"
            "original_response": response.strip(),
            "chosen_response": correction.strip() if (rating == "negative" and correction) else response.strip(),
            "rejected_response": response.strip() if rating == "negative" else "",
        }
        with open(PREFERENCES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Also log as an episodic memory event
        feedback_label = "Positive Thumbs-Up" if rating == "positive" else "Correction Recorded"
        self.log_episode(
            event=f"User Feedback: {feedback_label}",
            category="alignment",
            details=f"Prompt: {prompt[:60]}...",
        )
        return "Feedback recorded and queued for continuous learning alignment!"

    def get_preference_stats(self) -> Dict[str, int]:
        if not os.path.exists(PREFERENCES_FILE):
            return {"positive": 0, "negative": 0, "total": 0}
        pos = 0
        neg = 0
        try:
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("rating") == "positive":
                        pos += 1
                    elif item.get("rating") == "negative":
                        neg += 1
        except Exception:
            pass
        return {"positive": pos, "negative": neg, "total": pos + neg}

    # =========================================================================
    # 5. Lifelong Error Solutions
    # =========================================================================

    def save_solution(self, issue: str, resolution: str, root_cause: str = "", tags: Optional[List[str]] = None) -> str:
        solutions = self.load_solutions()
        sol_id = f"sol_{len(solutions) + 1:03d}"
        new_entry = {
            "id": sol_id,
            "issue": issue.strip(),
            "root_cause": root_cause.strip(),
            "resolution": resolution.strip(),
            "tags": tags or ["troubleshooting", "learning"],
        }
        solutions.append(new_entry)
        os.makedirs(os.path.dirname(SOLUTIONS_FILE), exist_ok=True)
        with open(SOLUTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(solutions, f, indent=2, ensure_ascii=False)
        return f"Stored solution [{sol_id}]: {issue}"

    def load_solutions(self) -> List[Dict[str, Any]]:
        if not os.path.exists(SOLUTIONS_FILE):
            return []
        try:
            with open(SOLUTIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def search_solutions(self, query: str) -> List[Dict[str, Any]]:
        solutions = self.load_solutions()
        words = query.lower().split()
        matches = []
        for sol in solutions:
            text = f"{sol.get('issue', '')} {sol.get('root_cause', '')} {sol.get('resolution', '')} {' '.join(sol.get('tags', []))}".lower()
            if any(w in text for w in words):
                matches.append(sol)
        return matches

    def get_solution_summary(self) -> str:
        solutions = self.load_solutions()
        if not solutions:
            return ""
        lines = ["=== LIFELONG ERROR & SOLUTION MEMORY (HOW ISSUES WERE RESOLVED) ==="]
        for s in solutions[-5:]:
            lines.append(f"• Issue: {s['issue']}")
            lines.append(f"  Fix: {s['resolution']}")
        return "\n".join(lines)

    # =========================================================================
    # 6. Profile & Custom Notes
    # =========================================================================

    def remember(self, key: str, value: str) -> str:
        """Stores custom persistent fact or preference."""
        clean_k = key.strip()
        clean_v = value.strip()
        if not self.data.get("custom_notes"):
            self.data["custom_notes"] = {}
        self.data["custom_notes"][clean_k] = clean_v
        self._save()
        self.log_episode(f"Learned preference: {clean_k} = {clean_v}", category="preference")
        return f"Saved to memory: '{clean_k}' = '{clean_v}'"

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
        """Generates dynamic memory summary for LLM prompt."""
        lines = []

        profile = self.data.get("discovered_profile", {})
        for k, v in profile.items():
            readable_k = k.replace("_", " ").title()
            if isinstance(v, list):
                lines.append(f"{readable_k}: {', '.join(str(x) for x in v)}")
            else:
                lines.append(f"{readable_k}: {v}")

        if self.data.get("latest_company"):
            lines.append(f"Current / Latest Company: {self.data['latest_company']}")

        timeline = self.data.get("career_progression", [])
        if timeline:
            lines.append("Chronological Career Progression:")
            for item in timeline:
                extra = []
                if item.get("ctc"):
                    extra.append(f"CTC: {item['ctc']}")
                if item.get("experience"):
                    extra.append(f"Exp: {item['experience']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                lines.append(f"  • {item['date']}: {item['company']}{extra_str}")

        notes = self.data.get("custom_notes", {})
        if notes:
            lines.append("User Notes & Preferences:")
            for k, v in notes.items():
                lines.append(f"  • {k}: {v}")

        # Include Procedural Runbooks if available
        procs = self.load_procedures()
        if procs:
            lines.append("Saved Engineering Workflows & Runbooks:")
            for p in list(procs.values())[:3]:
                lines.append(f"  • [{p['name']}]: {' -> '.join(p['steps'][:3])}")

        return "\n".join(lines) if lines else "No profile records discovered yet. Ingest emails or documents to learn."

    def sync_from_corpus(self, corpus_path: str = "data/emails_corpus.jsonl") -> Dict[str, Any]:
        """Scans email corpus dynamically to extract company transitions, CTC, education."""
        if not os.path.exists(corpus_path):
            return self.data

        pattern_company = re.compile(
            r"(?:Current Company|Current Organization|Current Employer)\s*[:\-]\s*([^\n\r,]+)",
            re.IGNORECASE,
        )
        pattern_exp = re.compile(r"(?:Total(?: IT)? Experience)\s*[:\-]\s*([^\n\r]+)", re.IGNORECASE)
        pattern_ctc = re.compile(r"(?:Current CTC|CTC \(Fixed \+ Variable\)|Fixed CTC)\s*[:\-]\s*([^\n\r]+)", re.IGNORECASE)
        pattern_loc = re.compile(r"(?:Current Location)\s*[:\-]\s*([^\n\r]+)", re.IGNORECASE)

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

                m_loc = pattern_loc.search(body)
                if m_loc:
                    val = clean_entity_value(m_loc.group(1))
                    if val and "location" not in discovered_info:
                        discovered_info["location"] = val

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
            detected_events.sort(key=lambda x: x["timestamp"])
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
    print("=== DYNAMICALLY DISCOVERED COGNITIVE MEMORY ===")
    print(mgr.get_summary())
    print("\n=== RECENT EPISODIC TIMELINE ===")
    print(mgr.get_recent_episodes_summary())
    print("\n=== HUMAN PREFERENCE STATS ===")
    print(mgr.get_preference_stats())
