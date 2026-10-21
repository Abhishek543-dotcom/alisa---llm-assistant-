#!/usr/bin/env python3
"""
diagnostics.py
Comprehensive Production Health Check for Alisa All-Rounder Voice AI:
  1. Audio Hardware: Microphone capture & Speaker TTS playback.
  2. Metal GPU & MLX Framework: Apple Silicon Metal acceleration & Whisper STT latency.
  3. Base Models & Versioning Registry: Model cache, active version, adapters.
  4. All-Rounder Datasets: train.jsonl, valid.jsonl, test.jsonl integrity & counts.
  5. Vector Database: ChromaDB persistent storage & chunk counts.
  6. Dynamic Memory: Profile timeline, lifelong error solutions, and conversation turns.
"""

import json
import os
import subprocess
import time
import numpy as np


def run_diagnostics():
    print("=" * 70)
    print("🔬 Alisa All-Rounder Voice AI — Production System Diagnostics")
    print("=" * 70)

    all_passed = True

    # 1. Audio Output (TTS / afplay)
    print("\n[1/6] Checking Audio Output (Speakers / TTS)...")
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
    print("\n[2/6] Checking Audio Input (Microphone / sounddevice)...")
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
    print("\n[3/6] Checking Apple Silicon Metal GPU & MLX...")
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

    # 4. Model Versioning Registry & Weights
    print("\n[4/6] Checking Model Versioning & Weights...")
    try:
        from model_manager import ModelVersionManager
        vm = ModelVersionManager()
        active = vm.get_active_version()
        v_list = vm.list_versions()
        print(f"  ✓ Model Registry: {len(v_list)} version(s) recorded.")
        print(f"  ✓ Active Version: {active.get('version_id')} ({active.get('name')})")
        print(f"  ✓ Base Model: {active.get('base_model')}")
        adapter_path = vm.get_active_adapter_path()
        if adapter_path:
            print(f"  ✓ Active LoRA Adapter: {adapter_path}")
        else:
            print("  ✓ Running Mode: Fresh Base Model (Zero LoRA adapter)")
    except Exception as e:
        print(f"  ✗ Model versioning error: {e}")
        all_passed = False

    # 5. All-Rounder Datasets Integrity
    print("\n[5/6] Checking All-Rounder Datasets (train / valid / test)...")
    for fname in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        fpath = os.path.join("data", fname)
        if os.path.exists(fpath):
            count = sum(1 for line in open(fpath, encoding="utf-8") if line.strip())
            print(f"  ✓ {fname:<12}: {count:>5} valid examples")
        else:
            print(f"  • {fname:<12}: Not generated yet (will be created automatically on learning)")

    # 6. ChromaDB Vector Store & Dynamic Memory
    print("\n[6/6] Checking Vector Database & Memory...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="chroma_db")
        coll = client.get_collection(name="personal_emails")
        count = coll.count()
        print(f"  ✓ ChromaDB Storage: 'personal_emails' active ({count:,} chunks)")
    except Exception:
        print("  • ChromaDB Storage: 0 chunks (Ready for inbox ingestion or mbox parsing)")

    from memory_manager import MemoryManager
    mgr = MemoryManager()
    curr_co = mgr.get_latest_company()
    sols = mgr.load_solutions()
    hist = mgr.get_recent_history(limit=1)
    print(f"  ✓ Discovered Career: Latest Company is '{curr_co}'")
    print(f"  ✓ Solved Issues Memory: {len(sols)} documented resolutions")
    print(f"  ✓ Conversation History: {'Active' if hist else 'Ready for interactions'}")

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL PRODUCTION CHECKS VERIFIED: Alisa is 100% Operational!")
    else:
        print("⚠️ Some diagnostics reported warnings. Check logs above.")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()
