#!/usr/bin/env python3
"""
proactive_agent.py
Autonomous Proactive Intelligence & Daily Briefing Agent for Alisa:
  - Generates comprehensive Morning/Evening executive briefings for Abhishek.
  - Monitors system health anomalies (critical battery, low storage, high RAM load).
  - Summarizes active model version, ChromaDB knowledge stats, and episodic milestones.
  - Can vocalize briefings via macOS native Samantha TTS or render in the terminal.
"""

import argparse
import datetime
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from memory_manager import MemoryManager
from model_manager import ModelVersionManager
from system_tools import SystemTools
from voice_engine import VoiceEngine


class ProactiveAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.tools = SystemTools()
        self.version_mgr = ModelVersionManager()

    def get_greeting(self) -> str:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning, Abhishek!"
        elif 12 <= hour < 17:
            return "Good afternoon, Abhishek!"
        elif 17 <= hour < 22:
            return "Good evening, Abhishek!"
        else:
            return "Hello Abhishek! Burning the midnight oil?"

    def check_anomalies(self) -> List[str]:
        """Detects system anomalies requiring Abhishek's immediate attention."""
        alerts = []
        try:
            # Battery check
            batt_status = self.tools.get_battery_status()
            if "on battery" in batt_status.lower():
                import re
                m = re.search(r"(\d+)%", batt_status)
                if m and int(m.group(1)) <= 20:
                    alerts.append(f"⚠️ Low Battery Warning: Battery is at {m.group(1)}%. Connect your charger.")

            # Disk check
            stats = self.tools.get_system_stats()
            disk_free = stats.get("disk_free_gb", 100)
            if disk_free < 15.0:
                alerts.append(f"⚠️ Low Disk Storage: Only {disk_free}GB free space remaining on disk.")

            # RAM check
            ram_pct = stats.get("ram_pct", 0)
            if ram_pct >= 90:
                alerts.append(f"⚠️ High Memory Load: RAM usage is at {ram_pct}%. Consider closing heavy background apps.")
        except Exception:
            pass
        return alerts

    def generate_briefing(self) -> Dict[str, str]:
        """
        Synthesizes a full executive briefing combining system health,
        AI engine version, recent accomplishments, and current career status.
        """
        greeting = self.get_greeting()
        date_str = datetime.datetime.now().strftime("%A, %B %d, %Y (%I:%M %p)")

        # System metrics
        stats = self.tools.get_system_stats()
        batt = self.tools.get_battery_status()
        alerts = self.check_anomalies()

        # Engine metrics
        cur_v = self.version_mgr.get_active_version()
        v_id = cur_v.get("version_id", "v0")
        base_m = cur_v.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct").split("/")[-1]

        # Knowledge stats
        sols_count = len(self.memory.load_solutions())
        latest_co = self.memory.get_latest_company()

        # Vector chunk count
        try:
            import chromadb
            client = chromadb.PersistentClient(path="chroma_db")
            chunks = client.get_collection("personal_emails").count()
        except Exception:
            chunks = 0

        # Construct Vocal Script (Clean, conversational, Alexa-like)
        vocal_lines = [
            f"{greeting}",
            f"Here is your executive briefing for {datetime.datetime.now().strftime('%A')}.",
            f"Your system is running smoothly. Battery is {batt.replace('Battery is at', 'at')}.",
            f"RAM usage is {stats.get('ram_pct')}% with {stats.get('disk_free_gb')}GB free storage.",
            f"Your AI engine is running {base_m} at Model Version {v_id}.",
        ]

        if alerts:
            vocal_lines.append(f"Notice: {alerts[0]}")

        vocal_lines.append("I am ready for your coding tasks, queries, or voice commands.")
        vocal_script = " ".join(vocal_lines)

        # Construct Visual Dashboard
        dash_lines = [
            "=" * 70,
            f"🌅 ALISA EXECUTIVE BRIEFING — {greeting.upper()}",
            f"   {date_str}",
            "=" * 70,
            "\n💻 SYSTEM HEALTH & HARDWARE:",
            f"  • Power:       {batt}",
            f"  • Memory:      {stats.get('ram_pct')}% RAM ({stats.get('ram_used_gb')}GB / {stats.get('ram_total_gb')}GB used)",
            f"  • CPU Load:    {stats.get('cpu_pct')}%",
            f"  • Storage:     {stats.get('disk_free_gb')}GB Free / {stats.get('disk_total_gb')}GB Total",
        ]

        if alerts:
            dash_lines.append("\n⚠️ SYSTEM ALERTS:")
            for a in alerts:
                dash_lines.append(f"  {a}")

        dash_lines.extend([
            "\n🧠 AI BRAIN & MEMORY STATE:",
            f"  • Base Model:      {base_m} (Apple Silicon Metal GPU)",
            f"  • Active Version:  {v_id} ({cur_v.get('name', 'N/A')})",
            f"  • Vector Database: {chunks:,} chunks indexed in ChromaDB",
            f"  • Career Anchor:   {latest_co}",
            f"  • Error Memory:    {sols_count} lifelong troubleshooting solutions known",
        ])

        # Recent episodic diary
        episodes_text = self.memory.get_recent_episodes_summary(days=2)
        if episodes_text and "No episodic" not in episodes_text:
            dash_lines.append("\n📅 RECENT MILESTONES & EPISODIC DIARY:")
            for line in episodes_text.splitlines():
                dash_lines.append(f"  {line}")

        dash_lines.extend([
            "\n" + "=" * 70,
            "✨ Alisa is on standby. Say 'Alisa' or type your question below.",
            "=" * 70,
        ])

        visual_dashboard = "\n".join(dash_lines)

        # Log briefing generation as an episodic memory event
        self.memory.log_episode(
            event="Daily Executive Briefing Generated",
            category="briefing",
            details=f"System: RAM {stats.get('ram_pct')}%, Battery {batt}",
        )

        return {
            "vocal_script": vocal_script,
            "visual_dashboard": visual_dashboard,
        }


def main():
    parser = argparse.ArgumentParser(description="Alisa Proactive Intelligence & Daily Briefing Agent")
    parser.add_argument("--speak", action="store_true", help="Vocalize briefing via Samantha TTS")
    parser.add_argument("--silent", action="store_true", help="Do not speak, output text only")
    args = parser.parse_args()

    agent = ProactiveAgent()
    briefing = agent.generate_briefing()

    print(briefing["visual_dashboard"])

    if args.speak and not args.silent:
        voice = VoiceEngine(tts_enabled=True)
        print("\n[🔊 Speaking Executive Briefing...]")
        voice.speak(briefing["vocal_script"], non_blocking=False)


if __name__ == "__main__":
    main()
