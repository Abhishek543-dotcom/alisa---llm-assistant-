#!/usr/bin/env python3
"""
voice_engine.py
Ultra-low resource Voice System (TTS + STT) optimized for Apple Silicon macOS:
  - Text-to-Speech (TTS): Uses macOS native `/usr/bin/say`. 0 RAM, 0 CPU overhead, crystal-clear, non-blocking.
  - Speech-to-Text (STT): Uses `mlx-whisper` (tiny model, ~75MB) running directly on Apple Silicon Metal GPU.
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Optional


class VoiceEngine:
    def __init__(
        self,
        tts_enabled: bool = True,
        voice_name: str = "Samantha",
        stt_model: str = "mlx-community/whisper-tiny",
    ):
        self.tts_enabled = tts_enabled
        self.voice_name = voice_name
        self.stt_model = stt_model
        self._current_process: Optional[subprocess.Popen] = None

    def toggle_tts(self) -> str:
        self.tts_enabled = not self.tts_enabled
        state = "ON" if self.tts_enabled else "OFF"
        return f"Voice speech output (TTS) turned {state}."

    def speak(self, text: str, non_blocking: bool = True) -> None:
        """Speaks text using macOS native /usr/bin/say via stdin pipe with 0 Python RAM overhead."""
        if not self.tts_enabled or not text or not text.strip():
            return

        # Stop any currently speaking speech before starting new speech
        self.stop_speaking()

        # Clean up text for natural human-like speech
        clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [title](link) -> title
        clean_text = re.sub(r"https?://\S+", "", clean_text)          # Strip URLs
        clean_text = re.sub(r"[\*\_#`\>•~|]", " ", clean_text)         # Strip markdown symbols
        clean_text = re.sub(r"^\s*[-\*•]\s*", "", clean_text)          # Strip leading list bullets
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            return

        # Truncate overly long speech to prevent runaway talking (limit ~600 chars at sentence end)
        if len(clean_text) > 600:
            cut = clean_text[:600]
            last_period = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
            if last_period > 200:
                clean_text = cut[: last_period + 1]
            else:
                clean_text = cut + "... and more."

        def _run():
            try:
                cmd = ["say"]
                if self.voice_name:
                    cmd.extend(["-v", self.voice_name])
                self._current_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._current_process.communicate(input=clean_text.encode("utf-8"), timeout=60)
            except Exception:
                try:
                    self._current_process = subprocess.Popen(["say"], stdin=subprocess.PIPE)
                    self._current_process.communicate(input=clean_text.encode("utf-8"), timeout=60)
                except Exception:
                    pass

        if non_blocking:
            t = threading.Thread(target=_run, daemon=True)
            t.start()
        else:
            _run()

    def stop_speaking(self) -> None:
        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass
        self._current_process = None

    def record_and_transcribe(self, duration_seconds: int = 5) -> str:
        """
        Records audio from the microphone using sounddevice and transcribes
        with mlx-whisper (tiny model) on Apple Silicon Metal.
        """
        # Mute speech before recording so it doesn't transcribe itself
        self.stop_speaking()

        try:
            import sounddevice as sd
            import soundfile as sf
            import mlx_whisper
        except ImportError as e:
            return f"[Voice Error: Missing dependencies ({e})]"

        sample_rate = 16000
        print(f"\n🎤 Listening... Speak now ({duration_seconds}s) [Ctrl+C to stop recording]...")

        try:
            audio = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print(" Recording stopped.")
        except Exception as e:
            return f"[Microphone error: {e}]"

        print("⚡ Transcribing voice with MLX Whisper (in-memory, Metal GPU)...")
        try:
            # Flatten audio buffer to 1D float32 array (no ffmpeg or disk file needed!)
            audio_data = audio.flatten().astype("float32")
            res = mlx_whisper.transcribe(audio_data, path_or_hf_repo=self.stt_model)
            text = res.get("text", "").strip()
            return text
        except Exception as e:
            return f"[Transcription error: {e}]"


if __name__ == "__main__":
    v = VoiceEngine(tts_enabled=True)
    print("Testing macOS TTS...")
    v.speak("Hello Abhishek! Voice engine is operational with zero resource overhead.", non_blocking=False)
