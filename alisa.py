#!/usr/bin/env python3
"""
alisa.py
One-command launcher for Alisa — Abhishek's Autonomous Personal Voice AI Assistant.

Usage:
  python3 alisa.py          -> Launch interactive console with voice & wake-word support
  python3 alisa.py --listen -> Launch hands-free Alexa/Siri mode directly (say "Alisa")
"""

import sys
from chat_assistant import main

if __name__ == "__main__":
    main()
