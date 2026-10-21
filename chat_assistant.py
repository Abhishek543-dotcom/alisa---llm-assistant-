#!/usr/bin/env python3
"""
chat_assistant.py
Abhishek's Autonomous Personal AI Assistant:
  1. Zero Hardcoding: Learns everything dynamically from emails, documents, and dropped files.
  2. Autonomous Learning Engine: Ingests any file (.pdf, .txt, .csv, .json, .py, etc.) using /learn or inbox/.
  3. Dynamic ChromaDB Attach / Detach on demand (/attach, /detach, /rag on, /rag off).
  4. Ultra-Fast Voice Features: Text-to-Speech (macOS say) & Speech-to-Text (Apple Silicon MLX Whisper).
  5. Persistent Discovered Memory: Dynamically tracks chronological career progression and personal facts.
"""

import argparse
import os
import platform
import re
from typing import List

from auto_learner import AutoLearner, ingest_inbox_directory
from memory_manager import MemoryManager
from rag_engine import RAGEngine
from system_tools import SystemTools
from voice_engine import VoiceEngine

CONVERSATIONAL_GREETINGS = {
    "alisa",
    "hey alisa",
    "hi alisa",
    "hello alisa",
    "hello",
    "hi",
    "hey",
    "howdy",
    "greetings",
    "who are you",
    "what is your name",
    "what can you do",
    "help",
    "good morning",
    "good evening",
    "good afternoon",
}


def generate_mlx(
    model,
    tokenizer,
    messages: List[dict],
    max_tokens: int = 512,
    sampler=None,
    logits_processors=None,
) -> str:
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )


def generate_hf(model, tokenizer, device: str, messages: List[dict], max_tokens: int = 512) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            repetition_penalty=1.15,
            do_sample=True,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Abhishek's Autonomous AI Assistant with Voice & Dynamic Learning.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID")
    parser.add_argument("--adapter-dir", default="adapters/personal_lora", help="Trained LoRA adapter directory")
    parser.add_argument("--no-adapter", action="store_true", help="Run base model directly without LoRA adapter")
    parser.add_argument("--db-dir", default="chroma_db", help="ChromaDB directory")
    parser.add_argument("--no-rag", action="store_true", help="Start with ChromaDB detached")
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken voice output (macOS say is ON by default)")
    parser.add_argument("--listen", action="store_true", help="Start immediately in hands-free wake-word listening mode ('Alisa')")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model ID")
    parser.add_argument("--backend", choices=["auto", "mlx", "hf"], default="auto", help="Inference backend")
    parser.add_argument("--top-k", type=int, default=4, help="Number of emails/docs to retrieve via RAG")
    parser.add_argument("--query", default=None, help="Single question to ask (launches interactive shell if omitted)")
    args = parser.parse_args()
    args.model = re.sub(r"\s+", "", args.model.strip())

    backend = args.backend
    if backend == "auto":
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        try:
            import mlx_lm  # noqa: F401
            backend = "mlx" if is_apple_silicon else "hf"
        except ImportError:
            backend = "hf"

    # Initialize Modules (Zero Hardcoding)
    memory = MemoryManager()
    rag = RAGEngine(
        db_dir=args.db_dir,
        embedding_model=args.embedding_model,
        attached=(not args.no_rag),
    )
    learner = AutoLearner(db_dir=args.db_dir)
    voice = VoiceEngine(tts_enabled=(not args.no_voice))
    tools = SystemTools()

    use_adapter = not args.no_adapter and os.path.exists(args.adapter_dir)
    adapter_path = args.adapter_dir if use_adapter else None

    sampler = None
    logits_processors = None
    hf_device = None

    if backend == "mlx":
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        print(f"Loading MLX model {args.model}" + (f" with adapter {adapter_path}..." if adapter_path else "..."))
        model, tokenizer = load(args.model, adapter_path=adapter_path)
        sampler = make_sampler(temp=0.3, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.15)
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if hf_device in ("mps", "cuda") else torch.float32
        print(f"Loading HF model {args.model} on {hf_device}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(hf_device)
        if adapter_path:
            model = PeftModel.from_pretrained(base_model, adapter_path).to(hf_device)
        else:
            model = base_model

    def answer_question(user_query: str, speak_sync: bool = False) -> str:
        # Check system actions (volume, battery, apps, shell execution)
        system_action_res = tools.match_and_execute(user_query)
        if system_action_res:
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(system_action_res, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, system_action_res, source="voice" if speak_sync else "console")
            return system_action_res

        # Check conversational greetings & identity checks
        clean_q = user_query.strip().lower().rstrip("!?.")
        if clean_q in ("alisa", "hey alisa", "hi alisa", "hello alisa"):
            reply = "Hello Abhishek! I am Alisa, your personal assistant. I'm listening, how can I help you today?"
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(reply, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            return reply

        if clean_q in ("who are you", "what is your name", "what's your name", "introduce yourself", "tell me about yourself"):
            reply = "I am Alisa, your autonomous personal AI assistant. I know your career history, records, and documents, and I am here to assist you with anything you need!"
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(reply, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            return reply

        if clean_q in ("who am i", "what is my name", "what's my name"):
            curr_co = memory.get_latest_company()
            reply = f"You are Abhishek Tiwari, currently working at {curr_co}."
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(reply, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            return reply

        # Check if querying past error troubleshooting solutions
        if any(w in clean_q for w in ("how did we fix", "resolved", "solution for", "troubleshoot")) or (
            "error" in clean_q and any(w in clean_q for w in ("fix", "solve", "how", "what was", "remember"))
        ):
            matched_solutions = memory.search_solutions(clean_q)
            if matched_solutions:
                sol_lines = ["I found the past solution in my error-resolution memory:"]
                for s in matched_solutions[:2]:
                    sol_lines.append(f"• Issue: {s['issue']}")
                    sol_lines.append(f"  Root Cause: {s.get('root_cause', 'N/A')}")
                    sol_lines.append(f"  Fix: {s['resolution']}")
                reply = "\n".join(sol_lines)
                if voice.tts_enabled:
                    print("\n[🔊 Alisa Speaking...]")
                    voice.speak(reply, non_blocking=(not speak_sync))
                memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
                return reply

        if clean_q in CONVERSATIONAL_GREETINGS:
            email_context = ""
        elif rag.is_attached:
            email_context = rag.search_and_format(user_query, top_k=args.top_k)
        else:
            email_context = "ChromaDB vector search is currently detached. Use working memory below."

        dynamic_memory = memory.get_summary()
        learned_summary = learner.get_summary()
        solutions_memory = memory.get_solution_summary()

        system_content = (
            "Your name is Alisa. You are Abhishek Tiwari's dedicated, highly capable, and intelligent personal AI assistant.\n"
            "You are NOT Abhishek Tiwari; Abhishek Tiwari is your boss/creator, and you are speaking directly to him.\n"
            "You know everything about Abhishek: his career journey, background, preferences, skills, and documents.\n"
            "You can do anything he asks: answer personal or technical questions, analyze documents, reason, summarize records, and draft messages.\n"
            "Personality: Warm, confident, respectful, intelligent, concise, conversational, and direct (just like Alexa or Siri).\n"
            "Always address Abhishek directly. Answer accurately without unnecessary conversational fluff or preambles.\n\n"
            "=== DYNAMIC WORKING MEMORY (DISCOVERED ABOUT ABHISHEK - ZERO HARDCODING) ===\n"
            f"{dynamic_memory}\n\n"
            f"{learned_summary}\n\n"
            f"{solutions_memory}\n\n"
            "=== RETRIEVED EVIDENCE FROM ARCHIVES & DOCUMENTS ===\n"
            f"{email_context}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. DISTILL AND SYNTHESIZE: Extract the exact answers and facts. Do NOT simply copy-paste or regurgitate raw text.\n"
            "2. CHRONOLOGICAL RECENCY: When asked about current company or role, always refer to the newest chronological discovery in memory.\n"
            "3. HONESTY: If a specific fact is not discovered in memory or documents, say: 'I could not find that information in your records.' Never guess or hallucinate.\n"
            "4. EMAIL DRAFTING: Only format as an email if the user explicitly asks to draft/write an email. For all other queries, answer directly in normal sentences."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

        if backend == "mlx":
            reply = generate_mlx(
                model,
                tokenizer,
                messages,
                max_tokens=512,
                sampler=sampler,
                logits_processors=logits_processors,
            )
        else:
            reply = generate_hf(model, tokenizer, hf_device, messages, max_tokens=512)

        # Log interaction for lifelong continuity
        memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")

        # Trigger spoken speech if enabled
        if voice.tts_enabled:
            print("\n[🔊 Alisa Speaking...]")
            voice.speak(reply, non_blocking=(not speak_sync))
        return reply

    def handle_voice_query(cmd: str):
        print(f"\nYou (Voice) > {cmd}")
        ans = answer_question(cmd, speak_sync=True)
        print(f"\nAlisa > {ans}")

    if args.listen:
        voice.listen_for_wake_word(on_wake_detected=handle_voice_query)
        return

    if args.query:
        ans = answer_question(args.query, speak_sync=True)
        print(ans)
        return

    print("=" * 65)
    print("✨ Alisa — Abhishek's Autonomous Personal Voice AI Assistant")
    print("   • Wake Word: Say \"Alisa\" or \"Hey Alisa\" (macOS chime activation)")
    print(f"   • ChromaDB RAG: {'ATTACHED (' + str(rag.count_chunks()) + ' chunks)' if rag.is_attached else 'DETACHED'}")
    print(f"   • Voice Speech (TTS): {'ENABLED (Samantha)' if voice.tts_enabled else 'MUTED (Type /voice to enable)'}")
    print(f"   • Drop Folder: inbox/ (drop any file to learn autonomously)")
    print("=" * 65)
    print("   [Commands]")
    print("   /listen                  -> Hands-free continuous Alexa/Siri mode (Wake: 'Alisa')")
    print("   /speak                   -> Push-to-talk mic mode (MLX Whisper)")
    print("   /voice                   -> Toggle spoken voice output (TTS)")
    print("   /learn <file or note>    -> Ingest, index, and learn from any file or text")
    print("   /scan                    -> Ingest all files dropped in inbox/")
    print("   /attach | /detach        -> Attach or detach ChromaDB vector search")
    print("   /memory                  -> View dynamically discovered memory profile")
    print("   /remember <key>: <value> -> Add a custom fact to memory")
    print("   /help                    -> Show commands")
    print("   exit                     -> Quit Alisa")
    print("=" * 65)

    while True:
        try:
            q = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        lower_q = q.lower()

        # Voice STT
        if lower_q == "/speak":
            spoken_text = voice.record_and_transcribe(duration_seconds=5)
            if not spoken_text:
                print("[Voice] No speech detected.")
                continue
            if spoken_text.startswith("["):
                print(f"\n{spoken_text}")
                continue
            # If user said "Alisa, <question>", strip wake word for cleaner processing
            _, cmd = voice.strip_wake_word(spoken_text)
            final_query = cmd if cmd else spoken_text
            print(f"\nYou (Voice) > {final_query}")
            q = final_query

        # Hands-free Wake-Word mode
        elif lower_q in ("/listen", "/wake", "/handsfree"):
            voice.listen_for_wake_word(on_wake_detected=handle_voice_query)
            continue

        # Voice TTS toggle
        elif lower_q in ("/voice", "/voice on", "/voice off"):
            if "on" in lower_q:
                voice.tts_enabled = True
                print("\n[Voice] Voice speech output (TTS) turned ON.")
            elif "off" in lower_q:
                voice.tts_enabled = False
                print("\n[Voice] Voice speech output (TTS) turned OFF.")
            else:
                print(f"\n[Voice] {voice.toggle_tts()}")
            continue

        # Ingestion commands
        elif lower_q.startswith("/learn"):
            payload = q[len("/learn") :].strip()
            if not payload:
                print("Usage: /learn <path_to_file> OR /learn <any note or statement>")
                continue
            if os.path.exists(payload):
                print(f"\n[AutoLearner] Processing file: {payload}...")
                res = learner.learn_file(payload)
            else:
                print(f"\n[AutoLearner] Ingesting knowledge statement...")
                res = learner.learn_text(payload)
            print(res)
            continue

        elif lower_q in ("/scan", "/inbox"):
            print("\n[AutoLearner] Scanning 'inbox/' directory...")
            ingest_inbox_directory("inbox")
            continue

        # RAG Attach / Detach
        elif lower_q in ("/attach", "/rag on"):
            print(f"\n[RAG] {rag.attach()}")
            continue
        elif lower_q in ("/detach", "/rag off"):
            print(f"\n[RAG] {rag.detach()}")
            continue
        elif lower_q in ("/rag status", "/status"):
            print(f"\n[RAG] Attached: {rag.is_attached} | Indexed Chunks: {rag.count_chunks()} | DB: {args.db_dir}")
            continue

        # Memory commands
        elif lower_q == "/memory":
            print("\n=== DYNAMICALLY DISCOVERED MEMORY (0 HARDCODING) ===")
            print(memory.get_summary())
            print("\n=== LEARNED EXTERNAL KNOWLEDGE ===")
            print(learner.get_summary())
            continue

        elif lower_q.startswith("/remember"):
            rest = q[len("/remember") :].strip()
            if ":" in rest:
                k, v = rest.split(":", 1)
                msg = memory.remember(k, v)
            elif rest:
                msg = memory.remember("note", rest)
            else:
                msg = "Usage: /remember <key>: <value>"
            print(f"\n[Memory] {msg}")
            continue

        # Lifelong Solution & Error History
        elif lower_q in ("/solutions", "/errors", "/troubleshoot"):
            sols = memory.load_solutions()
            print("\n=== LIFELONG ERROR & SOLUTION MEMORY ===")
            if not sols:
                print("No solutions stored yet.")
            for s in sols:
                print(f"[{s['id']}] {s['issue']}")
                print(f"       Root Cause: {s.get('root_cause', 'N/A')}")
                print(f"       Resolution: {s['resolution']}\n")
            continue

        # Conversation History
        elif lower_q in ("/history", "/turns"):
            hist = memory.get_recent_history(limit=5)
            print("\n=== RECENT CONVERSATION HISTORY ===")
            if not hist:
                print("No prior conversation history recorded.")
            for h in hist:
                print(f"• [{h.get('timestamp', '')}] ({h.get('source', '')})")
                print(f"  User:  {h.get('query', '')}")
                print(f"  Alisa: {h.get('answer', '')[:100]}...\n")
            continue

        # System Controls
        elif lower_q in ("/battery", "/power"):
            print(f"\n[System] {tools.get_battery_status()}")
            continue

        elif lower_q.startswith("/volume"):
            arg = q[len("/volume") :].strip()
            if arg:
                try:
                    lvl = int(arg)
                    print(f"\n[System] {tools.set_volume(lvl)}")
                except ValueError:
                    print("Usage: /volume <0-100>")
            else:
                print(f"\n[System] {tools.get_volume()}")
            continue

        elif lower_q in ("/stats", "/system"):
            st = tools.get_system_stats()
            print("\n=== SYSTEM HEALTH & METRICS ===")
            print(f"• RAM Usage: {st.get('ram_pct')}% ({st.get('ram_used_gb')}GB / {st.get('ram_total_gb')}GB)")
            print(f"• CPU Load:  {st.get('cpu_pct')}%")
            print(f"• Free Disk: {st.get('disk_free_gb')}GB / {st.get('disk_total_gb')}GB")
            continue

        elif lower_q.startswith("/run "):
            bash_cmd = q[len("/run ") :].strip()
            ok, out = tools.run_bash_command(bash_cmd)
            prefix = "✓" if ok else "✗"
            print(f"\n[{prefix} Executed: {bash_cmd}]\n{out}")
            continue

        elif lower_q in ("/help", "/?"):
            print("\nAvailable Commands:")
            print("  /listen                  Hands-free continuous mode (say 'Alisa' to wake, 'stop' to sleep)")
            print("  /speak                   Speak question via microphone (MLX Whisper)")
            print("  /voice                   Toggle spoken voice output (macOS say)")
            print("  /battery                 Check current battery percentage and power source")
            print("  /volume [0-100]          Check or set macOS output volume")
            print("  /stats                   Check RAM, CPU, and Disk status")
            print("  /run <command>           Run a bash command directly")
            print("  /solutions               View lifelong error & troubleshooting memory")
            print("  /history                 View recent conversation history")
            print("  /learn <file or text>    Ingest and learn any file (.pdf, .txt, .csv) or note")
            print("  /scan                    Ingest all files placed in the inbox/ folder")
            print("  /attach | /detach        Attach or detach ChromaDB vector search")
            print("  /memory                  View discovered profile and learned knowledge")
            print("  /remember <key>: <value> Save a custom fact to memory")
            print("  exit                     Quit")
            continue

        print("\nAlisa >", answer_question(q))


if __name__ == "__main__":
    main()
