#!/usr/bin/env python3
"""
voice_engine.py
Ultra-low resource Voice System (TTS + STT) optimized for Apple Silicon macOS:
  - Text-to-Speech (TTS): Uses macOS native `/usr/bin/say`. 0 RAM, 0 CPU overhead, crystal-clear, non-blocking.
  - Speech-to-Text (STT): Uses `mlx-whisper` (tiny model, ~75MB) running directly on Apple Silicon Metal GPU.
"""

import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Optional, Tuple

WAKE_WORD_REGEX = re.compile(
    r"\b(?:hey|hi|hello|okay|ok)?[\s,]+(?:alisa|elisa|eliza|alissa|alyssa|aleesa|elysa|alisha|alice|alicia|aliza)\b|\b(?:alisa|elisa|eliza|alissa|alyssa|aleesa|elysa|alisha|alice|alicia|aliza)\b",
    re.IGNORECASE,
)

STOP_LISTENING_REGEX = re.compile(
    r"\b(?:stop\s+listening|go\s+to\s+sleep|sleep|stop\s+alisa|alisa\s+stop|exit\s+listening|pause\s+listening|shut\s+down)\b",
    re.IGNORECASE,
)


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
            res = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=self.stt_model,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            text = res.get("text", "").strip()
            return text
        except Exception as e:
            return f"[Transcription error: {e}]"


    def play_chime(self, sound: str = "Tink", block: bool = False) -> None:
        """Plays macOS native alert sounds (Tink, Pop, Ping, etc.) via afplay with zero RAM overhead."""
        sound_path = f"/System/Library/Sounds/{sound}.aiff"
        if not os.path.exists(sound_path):
            return
        try:
            if block:
                subprocess.run(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def strip_wake_word(self, text: str) -> Tuple[bool, str]:
        """
        Detects if text contains the wake word 'Alisa' (or variations).
        Returns (has_wake_word, remaining_command).
        """
        match = WAKE_WORD_REGEX.search(text)
        if not match:
            return False, text.strip()
        cmd = text[match.end() :].strip().lstrip(",.?! ")
        return True, cmd

    def listen_for_wake_word(
        self,
        on_wake_detected: Callable[[str], None],
        stop_event: Optional[threading.Event] = None,
        duration_per_chunk: float = 0.5,
    ) -> None:
        """
        Continuously listens for 'Alisa' in the background with near-zero idle CPU.
        Uses asynchronous queue streaming + RMS speech gating:
          - If room is quiet (RMS < threshold), skips Whisper (0.0% GPU).
          - If speech is detected, runs Whisper on Apple Silicon GPU (~80ms).
          - When 'Alisa' is spoken:
              1. Plays macOS wake chime (Tink)
              2. Listens for user query
              3. Invokes on_wake_detected(query)
        """
        try:
            import mlx_whisper
            import numpy as np
            import sounddevice as sd
        except ImportError as e:
            print(f"[Alisa Voice Error] Missing dependencies: {e}")
            return

        sample_rate = 16000
        # 0.5s chunks via background callback
        chunk_samples = int(duration_per_chunk * sample_rate)
        # 2.0s sliding window
        window_samples = sample_rate * 2
        buffer = np.zeros(window_samples, dtype=np.float32)

        print("\n" + "=" * 65)
        print("🎙️  Alisa Hands-Free Wake-Word Mode Activated")
        print("   • Wake Word: Say \"Alisa\" or \"Hey Alisa\"")
        print("   • Wake Sound: Active (macOS chime)")
        print("   • Press Ctrl+C anytime to stop listening")
        print("=" * 65)

        # Pre-warm MLX Whisper so it is hot in memory
        print("⚡ Warming up Whisper on Apple Silicon Metal GPU...")
        try:
            mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32),
                path_or_hf_repo=self.stt_model,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
        except Exception:
            pass

        # Calibrate ambient noise floor
        print("🔊 Calibrating room noise floor (0.5s)...")
        try:
            ambient_sample = sd.rec(int(0.5 * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()
            ambient_rms = float(np.sqrt(np.mean(ambient_sample**2)))
            energy_threshold = max(ambient_rms * 2.8, 0.028)
            print(f"✓ Noise calibration complete (Speech Threshold: {energy_threshold:.4f})")
        except Exception as e:
            energy_threshold = 0.030
            print(f"Warning: Noise calibration defaulted ({e})")

        print('\n👂 Alisa is listening for "Alisa"...')

        audio_q = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_q.put(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=audio_callback,
        )
        stream.start()

        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                chunk_flat = chunk.flatten().astype(np.float32)
                chunk_rms = float(np.sqrt(np.mean(chunk_flat**2)))

                # Update sliding buffer
                buffer = np.concatenate([buffer[len(chunk_flat) :], chunk_flat])

                # Energy gating: Skip GPU inference if quiet room noise
                if chunk_rms < energy_threshold:
                    continue

                # Speech energy detected: run Whisper on Apple Silicon GPU
                try:
                    res = mlx_whisper.transcribe(
                        buffer,
                        path_or_hf_repo=self.stt_model,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.6,
                    )
                    heard_text = res.get("text", "").strip()
                except Exception:
                    continue

                if not heard_text:
                    continue

                # Check if user spoke a command to stop listening
                if STOP_LISTENING_REGEX.search(heard_text):
                    stream.stop()
                    self.play_chime("Ping", block=True)
                    print('\n😴 Alisa going to sleep (Microphone OFF)...')
                    self.speak("Going to sleep now. You can turn listening back on anytime.", non_blocking=False)
                    break

                is_wake, command = self.strip_wake_word(heard_text)
                if not is_wake:
                    # Voice heard, but not wake-word
                    if len(heard_text) > 2:
                        print(f'\r[Heard: "{heard_text[:45]}"] (say "Alisa" to activate)...', end="", flush=True)
                    continue

                # Wake word matched!
                print("\n")
                # Temporarily stop stream to prevent feedback loops
                stream.stop()

                # Check if command was a stop command (e.g. "Alisa, stop listening" or "Alisa go to sleep")
                if command and STOP_LISTENING_REGEX.search(command):
                    self.play_chime("Ping", block=True)
                    print('\n😴 Alisa going to sleep (Microphone OFF)...')
                    self.speak("Going to sleep now. You can turn listening back on anytime.", non_blocking=False)
                    break

                # Play Alexa/Siri style wake chime
                self.play_chime("Tink", block=True)

                if command:
                    # User said "Alisa, what is my CTC?" in one breath
                    print(f'✨ [Alisa Activated!] Heard: "{command}"')
                else:
                    # User only said "Alisa" or "Hey Alisa" -> listen for command
                    print("✨ [Alisa Activated!] 🎤 Listening for your command (5s)...")
                    command = self.record_and_transcribe(duration_seconds=5)
                    self.play_chime("Ping", block=False)
                    if command.startswith("["):
                        print(f"{command}")
                        command = ""

                if command and STOP_LISTENING_REGEX.search(command):
                    self.play_chime("Ping", block=True)
                    print('\n😴 Alisa going to sleep (Microphone OFF)...')
                    self.speak("Going to sleep now. You can turn listening back on anytime.", non_blocking=False)
                    break

                if command:
                    on_wake_detected(command)

                # Drain backlog from queue & reset sliding buffer
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break
                buffer = np.zeros(window_samples, dtype=np.float32)

                print('\n👂 Alisa is listening for "Alisa"...')
                stream.start()

        except KeyboardInterrupt:
            print("\n⏹️ Stopped Alisa hands-free listening.")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


if __name__ == "__main__":
    v = VoiceEngine(tts_enabled=True)
    print("Testing macOS TTS & Wake Sound...")
    v.play_chime("Tink", block=True)
    v.speak("Hello Abhishek! I am Alisa, your personal assistant.", non_blocking=False)
