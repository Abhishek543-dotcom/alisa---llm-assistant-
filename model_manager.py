#!/usr/bin/env python3
"""
model_manager.py
Unified Base Model Downloader & Model Versioning / Rollback Engine for Alisa:
  1. Base Model Downloader:
     - Downloads Hugging Face / MLX models directly into local cache.
     - Progress indication, size reporting, and pre-cached detection.
     - Supports Qwen2.5 (0.5B, 1.5B, Coder), Llama-3.2 (1B, 3B), Whisper, and custom HF repos.
  2. Model Versioning & Rollback Registry:
     - Version v0: Guaranteed clean Fresh Base Model (untuned).
     - Version v1, v2, v3...: Incremental immutable versions created after each fine-tune or data insertion.
     - Rollback to previous version or rollback to fresh base model.
     - Active model pointer management (adapters/active and adapters/personal_lora).
     - Detailed metadata registry in data/model_registry.json.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


POPULAR_MODELS = {
    "1": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 0.5B Instruct",
        "size": "~1.0 GB",
        "description": "Ultra-fast, lowest memory (~1GB RAM). Default for MacBook Air / low resources.",
    },
    "2": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "size": "~3.1 GB",
        "description": "High intelligence, outstanding reasoning and instruction following (~3GB RAM).",
    },
    "3": {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "name": "Qwen 2.5 Coder 1.5B",
        "size": "~3.1 GB",
        "description": "Specialized coding, debugging, and terminal bash command assistant.",
    },
    "4": {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B Instruct",
        "size": "~2.5 GB",
        "description": "Meta's lightweight compact instruction model (~2.5GB RAM).",
    },
    "5": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B Instruct",
        "size": "~6.4 GB",
        "description": "High-powered 3B model for deeper reasoning (~6GB RAM).",
    },
    "6": {
        "id": "BAAI/bge-small-en-v1.5",
        "name": "BGE Small English v1.5",
        "size": "~133 MB",
        "description": "Embedding model for ChromaDB vector search RAG.",
    },
    "7": {
        "id": "mlx-community/whisper-tiny",
        "name": "Whisper Tiny (MLX)",
        "size": "~75 MB",
        "description": "Ultra-fast Apple Silicon Metal GPU Speech-to-Text model.",
    },
}

REGISTRY_FILE = "data/model_registry.json"
VERSIONS_DIR = "adapters/versions"
ACTIVE_ADAPTER_DIR = "adapters/active"
LEGACY_ADAPTER_DIR = "adapters/personal_lora"


# ==============================================================================
# 1. Base Model Downloader
# ==============================================================================

def clean_model_id(model_id: str) -> str:
    return re.sub(r"\s+", "", model_id.strip())


def download_base_model(model_id: str, verbose: bool = True) -> str:
    """
    Downloads any model repository from Hugging Face Hub using snapshot_download.
    Works seamlessly with Apple Silicon MLX and PyTorch Hugging Face.
    """
    repo_id = clean_model_id(model_id)
    if verbose:
        print(f"\n[Model Downloader] Initializing download for: {repo_id}")
        print("[Model Downloader] Checking local cache / connecting to Hugging Face Hub...")

    try:
        from huggingface_hub import snapshot_download
        allow_patterns = [
            "*.json",
            "model*.safetensors",
            "*.safetensors",
            "*.bin",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ]
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        if verbose:
            print(f"[Model Downloader] ✓ Successfully downloaded and verified: {repo_id}")
            print(f"[Model Downloader] Local cache path: {local_dir}")
        return local_dir
    except Exception as e:
        if verbose:
            print(f"[Model Downloader] ✗ Error downloading {repo_id}: {e}")
        raise


def interactive_model_downloader():
    """Interactive CLI menu to pick and download popular models or custom repo."""
    print("\n" + "=" * 65)
    print("📥 Alisa Base Model Downloader")
    print("=" * 65)
    print("Select a model to download and cache locally:\n")

    for k, v in POPULAR_MODELS.items():
        print(f"  [{k}] {v['name']} ({v['size']})")
        print(f"      ID: {v['id']}")
        print(f"      {v['description']}")

    print("  [8] Custom Hugging Face Model ID (Enter any repo)")
    print("  [9] Cancel / Return to Menu\n")

    choice = input("Enter choice [1-9]: ").strip()
    if choice in POPULAR_MODELS:
        target_id = POPULAR_MODELS[choice]["id"]
    elif choice == "8":
        target_id = input("\nEnter Hugging Face repo ID (e.g., Qwen/Qwen2.5-0.5B-Instruct): ").strip()
        if not target_id:
            print("Cancelled.")
            return
    elif choice in ("9", "q", "Q"):
        return
    else:
        print("Invalid choice.")
        return

    try:
        download_base_model(target_id, verbose=True)
        print("\nModel is ready for training and inference!")
    except Exception as e:
        print(f"\nFailed to download model: {e}")


# ==============================================================================
# 2. Model Versioning & Rollback Registry
# ==============================================================================

class ModelVersionManager:
    """
    Manages immutable model versions and instant rollback:
      - Version v0 is always the 'Fresh Base Model' (untuned).
      - Versions v1, v2, v3... are created after fine-tuning.
      - Maintains active version pointer and metadata.
    """

    def __init__(self, registry_file: str = REGISTRY_FILE, versions_dir: str = VERSIONS_DIR):
        self.registry_file = registry_file
        self.versions_dir = versions_dir
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.registry_file) or ".", exist_ok=True)
        self.registry = self._load_registry()

    def _default_registry(self) -> Dict[str, Any]:
        return {
            "active_version": "v0",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "versions": [
                {
                    "version_id": "v0",
                    "name": "Fresh Base Model",
                    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "adapter_path": None,
                    "is_fresh_base": True,
                    "train_samples": 0,
                    "valid_samples": 0,
                    "test_samples": 0,
                    "sources": ["base_weights"],
                    "metrics": {},
                    "notes": "Original untuned base model without LoRA adapters.",
                }
            ],
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                    if "versions" in reg and isinstance(reg["versions"], list):
                        return reg
            except Exception:
                pass
        reg = self._default_registry()
        self._save_registry(reg)
        return reg

    def _save_registry(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is None:
            data = self.registry
        data["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_active_version_id(self) -> str:
        return self.registry.get("active_version", "v0")

    def get_active_version(self) -> Dict[str, Any]:
        active_id = self.get_active_version_id()
        for v in self.registry.get("versions", []):
            if v["version_id"] == active_id:
                return v
        # Fallback to v0
        return self.registry["versions"][0]

    def get_active_adapter_path(self) -> Optional[str]:
        """Returns path to active adapter, or None if active version is fresh base (v0)."""
        active = self.get_active_version()
        if active.get("is_fresh_base") or not active.get("adapter_path"):
            return None
        adapter_path = active.get("adapter_path")
        if os.path.exists(adapter_path):
            return adapter_path
        # If specific version dir does not exist, check active link
        if os.path.exists(ACTIVE_ADAPTER_DIR):
            return ACTIVE_ADAPTER_DIR
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        return self.registry.get("versions", [])

    def prepare_new_version(self, base_model: str, notes: str = "") -> Tuple[str, str]:
        """
        Calculates next version number (v1, v2, v3...), prepares its directory,
        and returns (version_id, version_dir).
        """
        existing_versions = self.registry.get("versions", [])
        max_num = 0
        for v in existing_versions:
            vid = v.get("version_id", "")
            match = re.match(r"^v(\d+)$", vid)
            if match:
                max_num = max(max_num, int(match.group(1)))

        new_num = max_num + 1
        new_version_id = f"v{new_num}"
        version_dir = os.path.join(self.versions_dir, new_version_id)
        os.makedirs(version_dir, exist_ok=True)
        return new_version_id, version_dir

    def register_and_activate_version(
        self,
        version_id: str,
        version_dir: str,
        base_model: str,
        train_samples: int = 0,
        valid_samples: int = 0,
        test_samples: int = 0,
        sources: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Saves metadata for new version, links active pointers, and activates it.
        """
        parent_version = self.get_active_version_id()
        entry = {
            "version_id": version_id,
            "name": f"Fine-Tuned {version_id}",
            "base_model": clean_model_id(base_model),
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "adapter_path": version_dir,
            "is_fresh_base": False,
            "parent_version": parent_version,
            "train_samples": train_samples,
            "valid_samples": valid_samples,
            "test_samples": test_samples,
            "sources": sources or [],
            "metrics": metrics or {},
            "notes": notes or f"Incremental fine-tuning run {version_id}",
        }

        # Check if version already in list
        versions = self.registry.get("versions", [])
        versions = [v for v in versions if v["version_id"] != version_id]
        versions.append(entry)
        self.registry["versions"] = versions
        self.registry["base_model"] = clean_model_id(base_model)
        self.registry["active_version"] = version_id
        self._sync_adapter_pointers(version_dir)
        self._save_registry()
        return entry

    def _sync_adapter_pointers(self, target_dir: Optional[str]) -> None:
        """Updates active adapter pointer and backward-compatible personal_lora pointer."""
        for pointer_path in (ACTIVE_ADAPTER_DIR, LEGACY_ADAPTER_DIR):
            try:
                if os.path.islink(pointer_path):
                    os.unlink(pointer_path)
                elif os.path.isdir(pointer_path):
                    # If target is specified and directory is non-empty, copy files
                    if target_dir and os.path.exists(target_dir):
                        # Keep directory in sync by copying or symlinking
                        for item in os.listdir(target_dir):
                            s = os.path.join(target_dir, item)
                            d = os.path.join(pointer_path, item)
                            if os.path.isfile(s):
                                shutil.copy2(s, d)
                elif not os.path.exists(pointer_path) and target_dir:
                    try:
                        os.symlink(os.path.abspath(target_dir), pointer_path)
                    except Exception:
                        shutil.copytree(target_dir, pointer_path)
            except Exception as e:
                print(f"[ModelVersionManager] Note on pointer sync: {e}")

    def activate_version(self, version_id: str) -> Tuple[bool, str]:
        """Activates an existing version by ID."""
        target = None
        for v in self.registry.get("versions", []):
            if v["version_id"] == version_id:
                target = v
                break

        if not target:
            return False, f"Version '{version_id}' does not exist in registry."

        self.registry["active_version"] = version_id
        if target.get("is_fresh_base"):
            # Fresh base model (no adapter)
            self._save_registry()
            return True, f"Activated version {version_id} (Fresh Base Model without LoRA adapter)."

        adapter_path = target.get("adapter_path")
        if not adapter_path or not os.path.exists(adapter_path):
            return False, f"Adapter directory for {version_id} not found at {adapter_path}."

        self._sync_adapter_pointers(adapter_path)
        self._save_registry()
        return True, f"Successfully activated version {version_id} ({target.get('name')})."

    def rollback(self) -> Tuple[bool, str]:
        """Rolls back to the previous version before the currently active one."""
        versions = self.registry.get("versions", [])
        if len(versions) <= 1:
            return False, "No previous version available to roll back to. Currently at fresh base (v0)."

        active_id = self.get_active_version_id()
        active_idx = -1
        for idx, v in enumerate(versions):
            if v["version_id"] == active_id:
                active_idx = idx
                break

        if active_idx <= 0:
            # If at v0 or not found, cannot rollback further
            return False, f"Already at earliest version ({active_id}). Cannot roll back further."

        previous_version = versions[active_idx - 1]
        prev_id = previous_version["version_id"]
        ok, msg = self.activate_version(prev_id)
        if ok:
            return True, f"Rolled back from {active_id} to {prev_id} ({previous_version.get('name')})."
        return False, msg

    def rollback_to_fresh(self) -> Tuple[bool, str]:
        """Instantly rolls back to the clean Fresh Base Model (v0)."""
        return self.activate_version("v0")

    def print_version_table(self) -> None:
        """Pretty-prints all model versions."""
        active_id = self.get_active_version_id()
        print("\n" + "=" * 75)
        print("🏛️  Alisa Model Version Registry & Rollback Center")
        print("=" * 75)
        print(f"Active Base Model: {self.registry.get('base_model', 'Qwen/Qwen2.5-0.5B-Instruct')}")
        print(f"Current Active Version: {active_id}\n")

        print(f"{'Version':<10} {'Status':<12} {'Created':<20} {'Samples':<12} {'Notes'}")
        print("-" * 75)

        for v in self.registry.get("versions", []):
            vid = v["version_id"]
            status = "★ [ACTIVE]" if vid == active_id else "  [Ready]"
            created = v.get("created_at", "N/A")[:19]
            samples = f"{v.get('train_samples', 0)} tr"
            notes = v.get("notes", "")[:25]
            if v.get("is_fresh_base"):
                samples = "0 (Base)"
            print(f"{vid:<10} {status:<12} {created:<20} {samples:<12} {notes}")

        print("=" * 75)


def interactive_version_menu():
    """Interactive CLI menu for model versions and rollback."""
    mgr = ModelVersionManager()

    while True:
        mgr.print_version_table()
        print("Actions:")
        print("  [1] Rollback to Previous Version (vN-1)")
        print("  [2] Rollback to Fresh Base Model (v0 - untuned clean slate)")
        print("  [3] Switch / Activate Specific Version")
        print("  [4] Return to Main Menu\n")

        choice = input("Enter choice [1-4]: ").strip()

        if choice == "1":
            ok, msg = mgr.rollback()
            print(f"\n{msg}")
            input("Press Enter to continue...")
        elif choice == "2":
            confirm = input("Confirm rollback to Fresh Base Model v0? (y/N): ").strip()
            if confirm.lower() == "y":
                ok, msg = mgr.rollback_to_fresh()
                print(f"\n{msg}")
            input("Press Enter to continue...")
        elif choice == "3":
            target_vid = input("Enter Version ID to activate (e.g., v0, v1, v2): ").strip()
            if target_vid:
                ok, msg = mgr.activate_version(target_vid)
                print(f"\n{msg}")
            input("Press Enter to continue...")
        elif choice in ("4", "q", "Q"):
            break
        else:
            print("Invalid option.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Alisa Model Downloader & Versioning Manager")
    parser.add_argument("--download", help="Download and cache a specific Hugging Face model ID")
    parser.add_argument("--list", action="store_true", help="List all model versions")
    parser.add_argument("--active", action="store_true", help="Print active model version info")
    parser.add_argument("--activate", help="Activate a specific version ID (e.g., v0, v1)")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous version")
    parser.add_argument("--rollback-fresh", action="store_true", help="Rollback to fresh base model (v0)")
    parser.add_argument("--menu", action="store_true", help="Launch interactive version management menu")
    parser.add_argument("--download-menu", action="store_true", help="Launch interactive model downloader menu")
    args = parser.parse_args()

    mgr = ModelVersionManager()

    if args.download:
        download_base_model(args.download, verbose=True)
    elif args.download_menu:
        interactive_model_downloader()
    elif args.list:
        mgr.print_version_table()
    elif args.active:
        active = mgr.get_active_version()
        print(json.dumps(active, indent=2))
    elif args.activate:
        ok, msg = mgr.activate_version(args.activate)
        print(msg)
        sys.exit(0 if ok else 1)
    elif args.rollback:
        ok, msg = mgr.rollback()
        print(msg)
        sys.exit(0 if ok else 1)
    elif args.rollback_fresh:
        ok, msg = mgr.rollback_to_fresh()
        print(msg)
        sys.exit(0 if ok else 1)
    elif args.menu:
        interactive_version_menu()
    else:
        mgr.print_version_table()


if __name__ == "__main__":
    main()
