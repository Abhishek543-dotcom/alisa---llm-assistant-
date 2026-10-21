#!/usr/bin/env python3
"""
system_tools.py
Production-grade Agent Toolset & System Control for Alisa on macOS:
  1. macOS System Controls: Volume, Battery, Disk, RAM/CPU, App launcher.
  2. Safe Code & Shell Execution: Run bash commands, write code, read files.
  3. Action Intent Detection: Automatically executes system actions from natural language.
"""

import os
import re
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple


class SystemTools:
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    # =========================================================================
    # 1. macOS System Controls
    # =========================================================================

    def get_battery_status(self) -> str:
        """Returns current battery percentage and charging state via pmset."""
        try:
            res = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
            output = res.stdout.strip()
            # Extract percentage: e.g., 42%
            match = re.search(r"(\d+%)", output)
            charging_match = re.search(r"(charging|discharging|charged|AC Power)", output, re.IGNORECASE)
            pct = match.group(1) if match else "Unknown"
            state = charging_match.group(1) if charging_match else "on battery"
            return f"Battery is at {pct} ({state})."
        except Exception as e:
            return f"Unable to retrieve battery status: {e}"

    def get_volume(self) -> str:
        """Returns current macOS volume level (0-100)."""
        try:
            res = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            vol = res.stdout.strip()
            return f"Current system volume is {vol}%."
        except Exception as e:
            return f"Unable to read volume: {e}"

    def set_volume(self, level: int) -> str:
        """Sets macOS volume level (0 to 100)."""
        level = max(0, min(100, int(level)))
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                check=True,
                timeout=5,
            )
            return f"System volume set to {level}%."
        except Exception as e:
            return f"Failed to set volume: {e}"

    def mute_volume(self) -> str:
        """Mutes macOS audio output."""
        try:
            subprocess.run(
                ["osascript", "-e", "set volume with output muted"],
                check=True,
                timeout=5,
            )
            return "System volume muted."
        except Exception as e:
            return f"Failed to mute: {e}"

    def unmute_volume(self) -> str:
        """Unmutes macOS audio output."""
        try:
            subprocess.run(
                ["osascript", "-e", "set volume without output muted"],
                check=True,
                timeout=5,
            )
            return "System volume unmuted."
        except Exception as e:
            return f"Failed to unmute: {e}"

    def get_system_stats(self) -> Dict[str, Any]:
        """Returns RAM, CPU load, and Disk storage metrics."""
        stats = {}
        try:
            import psutil

            stats["ram_pct"] = psutil.virtual_memory().percent
            stats["ram_used_gb"] = round(psutil.virtual_memory().used / (1024**3), 1)
            stats["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
            stats["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        except ImportError:
            stats["ram_pct"] = "N/A"
            stats["cpu_pct"] = "N/A"

        try:
            total, used, free = shutil.disk_usage("/")
            stats["disk_free_gb"] = round(free / (1024**3), 1)
            stats["disk_total_gb"] = round(total / (1024**3), 1)
        except Exception:
            stats["disk_free_gb"] = "N/A"

        return stats

    def open_application(self, app_name: str) -> str:
        """Opens a macOS application using `open -a`."""
        app_clean = app_name.strip()
        common_aliases = {
            "chrome": "Google Chrome",
            "browser": "Google Chrome",
            "vscode": "Visual Studio Code",
            "code": "Visual Studio Code",
            "notes": "Notes",
            "terminal": "Terminal",
            "finder": "Finder",
            "slack": "Slack",
            "spotify": "Spotify",
        }
        resolved = common_aliases.get(app_clean.lower(), app_clean)
        try:
            subprocess.run(["open", "-a", resolved], check=True, timeout=5)
            return f"Opened application: {resolved}."
        except Exception as e:
            return f"Could not open '{resolved}': {e}"

    # =========================================================================
    # 2. Code & Shell Execution
    # =========================================================================

    def run_bash_command(self, command: str, timeout: int = 30) -> Tuple[bool, str]:
        """Runs a shell command safely inside the workspace."""
        # Block blatantly destructive commands
        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:"]
        for d in dangerous:
            if d in command:
                return False, f"Refused to execute unsafe command: {command}"

        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            combined = out if out else err
            if not combined:
                combined = f"Command finished with return code {res.returncode}."
            return (res.returncode == 0), combined
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds."
        except Exception as e:
            return False, f"Execution error: {e}"

    def write_code_file(self, relative_path: str, content: str) -> str:
        """Writes or creates a code file safely within workspace."""
        target_path = os.path.abspath(os.path.join(self.workspace_root, relative_path))
        if not target_path.startswith(self.workspace_root):
            return "Error: Cannot write outside workspace."

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created/updated file: {relative_path}"

    def read_code_file(self, relative_path: str, max_lines: int = 150) -> str:
        """Reads a file safely within workspace."""
        target_path = os.path.abspath(os.path.join(self.workspace_root, relative_path))
        if not target_path.startswith(self.workspace_root) or not os.path.exists(target_path):
            return f"Error: File '{relative_path}' not found."

        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(max_lines)]
            return "".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    # =========================================================================
    # 3. Intent Recognition & Dispatch
    # =========================================================================

    def match_and_execute(self, user_query: str) -> Optional[str]:
        """
        Parses common system intent patterns and executes them directly.
        Returns a response string if an action was matched and handled, or None.
        """
        q = user_query.strip().lower()

        # Volume control
        vol_set_match = re.search(r"(?:set|change|turn)?\s*(?:the)?\s*volume\s*(?:to)?\s*(\d+)", q)
        if vol_set_match:
            val = int(vol_set_match.group(1))
            return self.set_volume(val)

        if "mute" in q and "unmute" not in q:
            return self.mute_volume()
        if "unmute" in q:
            return self.unmute_volume()
        if any(x in q for x in ("what is the volume", "check volume", "get volume", "current volume")):
            return self.get_volume()

        # Battery status
        if any(x in q for x in ("battery", "battery level", "battery status", "charge", "power")):
            if any(x in q for x in ("what", "how", "check", "is", "tell", "status")):
                return self.get_battery_status()

        # System health / stats
        if any(x in q for x in ("system stats", "cpu usage", "ram usage", "memory usage", "disk space", "storage")):
            stats = self.get_system_stats()
            return (
                f"System Health: RAM is at {stats.get('ram_pct')}% ({stats.get('ram_used_gb')}GB / {stats.get('ram_total_gb')}GB). "
                f"CPU load is {stats.get('cpu_pct')}%. Free disk space: {stats.get('disk_free_gb')}GB."
            )

        # Open application
        open_match = re.search(r"open\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s+app|\s+application|$)", q)
        if open_match and not any(x in q for x in ("file", "code", "document", "email")):
            app = open_match.group(1).strip()
            if len(app) > 1 and app not in ("my", "your", "this", "that"):
                return self.open_application(app)

        # Shell command execution: e.g., "run command: git status" or "execute: ls -la"
        cmd_match = re.search(r"(?:run|execute)\s+(?:command|bash|shell)?\s*[:]\s*(.+)", user_query, re.IGNORECASE)
        if cmd_match:
            cmd = cmd_match.group(1).strip()
            ok, output = self.run_bash_command(cmd)
            status_str = "Success" if ok else "Failed"
            return f"[{status_str}] Output from `{cmd}`:\n{output}"

        return None


if __name__ == "__main__":
    tools = SystemTools()
    print("Testing SystemTools:")
    print(" • Battery:", tools.get_battery_status())
    print(" • Volume:", tools.get_volume())
    print(" • Stats:", tools.get_system_stats())
    print(" • Intent Match ('what is my battery level?'):", tools.match_and_execute("what is my battery level?"))
