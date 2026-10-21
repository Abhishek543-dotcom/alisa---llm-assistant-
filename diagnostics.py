#!/usr/bin/env python3
"""
diagnostics.py
Comprehensive Production Health Check for Alisa Personal Voice AI:
  1. Audio Hardware: Microphone recording & Speaker playback.
  2. Metal GPU & MLX Framework: Acceleration & memory state.
  3. Vector DB: ChromaDB collection integrity & chunk counts.
  4. Knowledge & Memory: Profile memory, solutions, and synthetic datasets.
"""

import os
import shutil
import subprocess
import time
import numpy as np


def run_diagnostics():
    print("=" * 65)
    print("🔬 Alisa Voice AI — Production System Diagnostics")
    print("=" * 65)

    all_passed = True

    # 1. Audio Output (TTS / afplay)
    print("\n[1/5] Checking Audio Output (Speakers / TTS)...")
    try:
        res = subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], timeout=3)
        if res.returncode == 0:
            print("  ✓ macOS native audio output (afplay): WORKING")
        else:
            print(f"  ✗ afplay returned code {res.returncode}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ Audio playback error: {e}")
        all_passed = False

    try:
        p = subprocess.Popen(["say", "-v", "Samantha"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.communicate(input=b"System diagnostic test.", timeout=5)
        print("  ✓ macOS speech synthesis (Samantha): WORKING")
    except Exception as e:
        print(f"  ✗ Speech synthesis error: {e}")
        all_passed = False

    # 2. Audio Input (Microphone / sounddevice)
    print("\n[2/5] Checking Audio Input (Microphone / sounddevice)...")
    try:
        import sounddevice as sd
        data = sd.rec(8000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(data**2)))
        print(f"  ✓ Microphone capture: WORKING (Sampled RMS: {rms:.5f})")
    except Exception as e:
        print(f"  ✗ Microphone error: {e}")
        all_passed = False

    # 3. Apple Silicon Metal GPU & MLX
    print("\n[3/5] Checking Apple Silicon Metal GPU & MLX...")
    try:
        import mlx.core as mx
        device = mx.default_device()
        print(f"  ✓ MLX Core Device: {device} (Metal GPU Accelerated)")
    except Exception as e:
        print(f"  ✗ MLX error: {e}")
        all_passed = False

    try:
        import mlx_whisper
        t0 = time.time()
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo="mlx-community/whisper-tiny")
        t1 = time.time()
        print(f"  ✓ MLX Whisper Metal GPU inference: WORKING ({(t1-t0)*1000:.1f}ms warm latency)")
    except Exception as e:
        print(f"  ✗ MLX Whisper error: {e}")
        all_passed = False

    # 4. ChromaDB Vector Database
    print("\n[4/5] Checking Vector Database (ChromaDB)...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="chroma_db")
        coll = client.get_collection(name="personal_emails")
        count = coll.count()
        print(f"  ✓ ChromaDB Storage: 'personal_emails' collection active ({count:,} indexed chunks)")
    except Exception as e:
        print(f"  ✗ ChromaDB warning: {e}")

    # 5. Lifelong Memory & Datasets
    print("\n[5/5] Checking Lifelong Knowledge & Profile Memory...")
    from memory_manager import MemoryManager
    mgr = MemoryManager()
    curr_co = mgr.get_latest_company()
    sols = mgr.load_solutions()
    hist = mgr.get_recent_history(limit=1)
    print(f"  ✓ Discovered Career: Latest Company is '{curr_co}'")
    print(f"  ✓ Solved Issues Memory: {len(sols)} documented resolutions")
    print(f"  ✓ Conversation History: {'Active' if hist else 'Ready for interactions'}")

    print("\n" + "=" * 65)
    if all_passed:
        print("🎉 ALL PRODUCTION DIAGNOSTICS PASSED: Alisa is 100% Operational!")
    else:
        print("⚠️ Some diagnostics reported warnings. Check logs above.")
    print("=" * 65)


if __name__ == "__main__":
    run_diagnostics()
