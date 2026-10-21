#!/usr/bin/env python3
"""
chat_assistant.py
Abhishek's Autonomous All-Rounder Personal AI Assistant:
  1. Multi-Tier Cognitive Brain:
     - All-Rounder Persona: Software engineering, coding, debugging, reasoning, macOS system controls,
       productivity, and deep personal memory synthesis.
     - Human Feedback Loop: Reinforce good answers (/good) or record corrections (/bad <correction>).
     - Episodic Timeline & Diary: Daily milestones and recap (/episodes, /diary).
     - Procedural Runbooks: Reusable workflows and execution procedures (/workflow, /workflows).
     - Proactive Executive Briefings: Daily morning/evening briefing (/briefing).
  2. Automated Model Versioning & Live Hot-Swap:
     - Detects active model version (v0 = Fresh Base, v1, v2... = Fine-tuned LoRA).
     - Live console rollback (/rollback, /fresh, /switch <vX>).
     - Standalone base model downloader (/download <model_id>).
  3. Dynamic Vector Search (RAG): Attach or detach ChromaDB on demand (/attach, /detach).
  4. Autonomous Learning Engine: Ingests any document, code, or data file from inbox/ (/learn, /scan).
  5. Voice Capabilities: Hands-free wake word ("Alisa"), push-to-talk mic (/speak), and native macOS TTS.
"""

import argparse
import os
import platform
import re
from typing import List, Optional

from auto_learner import AutoLearner, ingest_inbox_directory
from memory_manager import MemoryManager
from model_manager import ModelVersionManager, download_base_model
from proactive_agent import ProactiveAgent
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
    parser = argparse.ArgumentParser(description="Abhishek's All-Rounder Autonomous AI Assistant.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID")
    parser.add_argument("--adapter-dir", default=None, help="Explicit LoRA adapter path (if omitted, uses active version)")
    parser.add_argument("--no-adapter", action="store_true", help="Force fresh base model without LoRA adapter")
    parser.add_argument("--db-dir", default="chroma_db", help="ChromaDB directory")
    parser.add_argument("--no-rag", action="store_true", help="Start with ChromaDB detached")
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken voice output (TTS)")
    parser.add_argument("--listen", action="store_true", help="Start immediately in hands-free wake-word listening mode ('Alisa')")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model ID")
    parser.add_argument("--backend", choices=["auto", "mlx", "hf"], default="auto", help="Inference backend")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve via RAG")
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
    version_mgr = ModelVersionManager()
    memory = MemoryManager()
    rag = RAGEngine(
        db_dir=args.db_dir,
        embedding_model=args.embedding_model,
        attached=(not args.no_rag),
    )
    learner = AutoLearner(db_dir=args.db_dir)
    voice = VoiceEngine(tts_enabled=(not args.no_voice))
    tools = SystemTools()
    proactive = ProactiveAgent()

    # Determine Adapter
    if args.no_adapter:
        active_adapter_path = None
    elif args.adapter_dir:
        active_adapter_path = args.adapter_dir if os.path.exists(args.adapter_dir) else None
    else:
        active_adapter_path = version_mgr.get_active_adapter_path()

    active_ver = version_mgr.get_active_version()

    sampler = None
    logits_processors = None
    hf_device = None
    model = None
    tokenizer = None

    # Track last turn for human reinforcement feedback (/good, /bad)
    last_user_query = ""
    last_assistant_reply = ""

    def load_active_model(adapter_path_to_load: Optional[str]):
        nonlocal model, tokenizer, sampler, logits_processors, hf_device
        if backend == "mlx":
            from mlx_lm import load
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            adapter_label = f"adapter '{adapter_path_to_load}'" if adapter_path_to_load else "Fresh Base Model (No Adapter)"
            print(f"[Engine] Loading MLX {args.model} with {adapter_label}...")
            model, tokenizer = load(args.model, adapter_path=adapter_path_to_load)
            sampler = make_sampler(temp=0.3, top_p=0.9)
            logits_processors = make_logits_processors(repetition_penalty=1.15)
        else:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if hf_device in ("mps", "cuda") else torch.float32
            print(f"[Engine] Loading HF {args.model} on {hf_device}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(hf_device)
            if adapter_path_to_load:
                model = PeftModel.from_pretrained(base_model, adapter_path_to_load).to(hf_device)
            else:
                model = base_model

    # Initial Model Load
    load_active_model(active_adapter_path)

    def answer_question(user_query: str, speak_sync: bool = False) -> str:
        nonlocal last_user_query, last_assistant_reply
        last_user_query = user_query.strip()

        # Check system actions (volume, battery, apps, shell execution)
        system_action_res = tools.match_and_execute(user_query)
        if system_action_res:
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(system_action_res, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, system_action_res, source="voice" if speak_sync else "console")
            last_assistant_reply = system_action_res
            return system_action_res

        # Check conversational greetings & identity checks
        clean_q = user_query.strip().lower().rstrip("!?.")
        if clean_q in ("alisa", "hey alisa", "hi alisa", "hello alisa"):
            reply = "Hello Abhishek! I am Alisa, your all-rounder personal AI assistant. How can I help you today?"
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(reply, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            last_assistant_reply = reply
            return reply

        if clean_q in ("who are you", "what is your name", "what's your name", "introduce yourself", "tell me about yourself"):
            cur_v = version_mgr.get_active_version_id()
            reply = (
                f"I am Alisa, Abhishek Tiwari's autonomous all-rounder personal AI assistant (Running {args.model}, "
                f"Version: {cur_v}). I excel at software engineering, coding, debugging, document analysis, macOS system "
                f"controls, and maintaining your lifelong personal memory."
            )
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(reply, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            last_assistant_reply = reply
            return reply

        if clean_q in ("who am i", "what is my name", "what's my name"):
            curr_co = memory.get_latest_company()
            reply = f"You are Abhishek Tiwari, currently working at {curr_co}."
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak(reply, non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            last_assistant_reply = reply
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
                last_assistant_reply = reply
                return reply

        # Check if querying episodic timeline
        if any(w in clean_q for w in ("what did i do", "what did we do", "my milestones", "summary of my day", "what did i work on")):
            summary = memory.get_recent_episodes_summary(days=3)
            reply = f"Here is your recent episodic activity recap:\n\n{summary}"
            if voice.tts_enabled:
                print("\n[🔊 Alisa Speaking...]")
                voice.speak("Here is your recent activity recap.", non_blocking=(not speak_sync))
            memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
            last_assistant_reply = reply
            return reply

        if clean_q in CONVERSATIONAL_GREETINGS:
            email_context = ""
        elif rag.is_attached:
            email_context = rag.search_and_format(user_query, top_k=args.top_k)
        else:
            email_context = "Vector search is currently detached. Use working memory below."

        dynamic_memory = memory.get_summary()
        learned_summary = learner.get_summary()
        solutions_memory = memory.get_solution_summary()

        system_content = (
            "Your name is Alisa. You are Abhishek Tiwari's dedicated, highly capable, and intelligent all-rounder personal AI assistant.\n"
            "You are NOT Abhishek Tiwari; Abhishek Tiwari is your creator and boss, and you are speaking directly to him.\n"
            "You possess all-rounder mastery across: software engineering (Python, Bash, JS/TS, ML/AI), system architecture, "
            "technical debugging, logical reasoning, document analysis, and daily productivity.\n"
            "You know everything about Abhishek: his career journey, background, preferences, skills, and documents.\n"
            "Personality: Confident, intelligent, concise, conversational, and direct (just like Alexa or Siri).\n"
            "Always address Abhishek directly. Answer accurately without unnecessary conversational fluff or preambles.\n\n"
            "=== DYNAMIC WORKING MEMORY (DISCOVERED ABOUT ABHISHEK - ZERO HARDCODING) ===\n"
            f"{dynamic_memory}\n\n"
            f"{learned_summary}\n\n"
            f"{solutions_memory}\n\n"
            "=== RETRIEVED EVIDENCE FROM ARCHIVES & INBOX DOCUMENTS ===\n"
            f"{email_context}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. ALL-ROUNDER ASSISTANCE: Answer coding, technical, productivity, or general questions comprehensively with working code.\n"
            "2. CHRONOLOGICAL RECENCY: When asked about current company or role, always refer to the newest chronological discovery in memory.\n"
            "3. HONESTY: If a specific personal fact is not discovered in memory or documents, say: 'I could not find that information in your records.' Never guess or hallucinate.\n"
            "4. DIRECTNESS: Only format as an email if the user explicitly asks to draft/write an email. For all other queries, answer directly in normal sentences or code blocks."
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

        memory.log_interaction(user_query, reply, source="voice" if speak_sync else "console")
        last_assistant_reply = reply

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

    cur_vid = version_mgr.get_active_version_id()
    cur_v_info = version_mgr.get_active_version()

    print("=" * 70)
    print("✨ Alisa — Abhishek's Autonomous All-Rounder Personal AI Assistant")
    print(f"   • Model Engine: {args.model} (Metal GPU Accelerated)")
    print(f"   • Active Version: {cur_vid} ({cur_v_info.get('name', 'N/A')})")
    print(f"   • Wake Word: Say \"Alisa\" or \"Hey Alisa\" (macOS chime activation)")
    print(f"   • ChromaDB RAG: {'ATTACHED (' + str(rag.count_chunks()) + ' chunks)' if rag.is_attached else 'DETACHED'}")
    print(f"   • Voice Speech (TTS): {'ENABLED (Samantha)' if voice.tts_enabled else 'MUTED (/voice to toggle)'}")
    print(f"   • Inbox Drop Folder: inbox/ (drop any file to learn autonomously)")
    print("=" * 70)
    print("   [Commands]")
    print("   /briefing                -> Generate executive system & productivity briefing")
    print("   /good                    -> Record positive feedback on previous answer (RLHF)")
    print("   /bad <better_answer>     -> Correct previous answer for continuous alignment")
    print("   /episodes | /diary       -> View recent daily milestones and activity timeline")
    print("   /workflow [name]         -> View or save developer execution procedures")
    print("   /version | /models       -> View all model versions & active status")
    print("   /rollback                -> Rollback to previous model version (hot-swap)")
    print("   /fresh                   -> Rollback to Fresh Base Model (v0)")
    print("   /switch <vX>             -> Switch active model to version vX")
    print("   /download <model_id>     -> Download & cache base model from Hugging Face")
    print("   /listen                  -> Hands-free continuous Alexa/Siri mode")
    print("   /speak                   -> Push-to-talk mic mode (MLX Whisper)")
    print("   /voice                   -> Toggle spoken voice output (TTS)")
    print("   /learn <file or text>    -> Ingest, index, and learn from any file or text")
    print("   /scan                    -> Ingest all new files placed in inbox/")
    print("   /attach | /detach        -> Attach or detach ChromaDB vector search")
    print("   /memory                  -> View dynamically discovered cognitive memory")
    print("   /remember <key>: <val>   -> Save a custom fact to memory")
    print("   exit                     -> Quit Alisa")
    print("=" * 70)

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

        # Proactive Executive Briefing
        if lower_q in ("/briefing", "/morning", "/report", "/summary"):
            b = proactive.generate_briefing()
            print("\n" + b["visual_dashboard"])
            if voice.tts_enabled:
                print("\n[🔊 Speaking Executive Briefing...]")
                voice.speak(b["vocal_script"], non_blocking=True)
            continue

        # Human Feedback & Alignment (/good, /bad)
        elif lower_q in ("/good", "/thumbsup", "/like", "/+1"):
            if not last_user_query or not last_assistant_reply:
                print("\n[Feedback] No prior interaction recorded in this session to evaluate.")
            else:
                msg = memory.record_feedback(last_user_query, last_assistant_reply, rating="positive")
                print(f"\n[Feedback] ✓ {msg}")
            continue

        elif lower_q.startswith("/bad") or lower_q in ("/thumbsdown", "/dislike", "/-1"):
            if not last_user_query or not last_assistant_reply:
                print("\n[Feedback] No prior interaction recorded in this session to evaluate.")
            else:
                correction = q[4:].strip() if lower_q.startswith("/bad") else ""
                msg = memory.record_feedback(last_user_query, last_assistant_reply, rating="negative", correction=correction)
                print(f"\n[Feedback] ✓ {msg}")
                if correction:
                    print(f"  Correction queued: '{correction[:80]}...'")
            continue

        elif lower_q in ("/feedback", "/preferences"):
            stats = memory.get_preference_stats()
            print("\n=== HUMAN PREFERENCE & ALIGNMENT STATS ===")
            print(f"• Positive Thumbs-Up: {stats['positive']}")
            print(f"• Corrected Answers:  {stats['negative']}")
            print(f"• Total Data Points:  {stats['total']}")
            continue

        # Episodic Memory Diary
        elif lower_q in ("/episodes", "/diary", "/timeline"):
            print("\n=== RECENT EPISODIC TIMELINE & MILESTONES ===")
            print(memory.get_recent_episodes_summary(days=5))
            continue

        # Procedural Workflows
        elif lower_q.startswith("/workflow"):
            arg = q[len("/workflow") :].strip()
            if not arg or arg.lower() in ("list", "all"):
                procs = memory.load_procedures()
                print("\n=== SAVED DEVELOPER WORKFLOWS & RUNBOOKS ===")
                if not procs:
                    print("No procedures saved yet. Create one with: /workflow <name> = <step1> | <step2> | <step3>")
                for p in procs.values():
                    print(f"\n• {p['name']} (Trigger: '{p.get('trigger', '')}')")
                    for i, step in enumerate(p.get("steps", [])):
                        print(f"  {i+1}. {step}")
            elif "=" in arg:
                # Creation: /workflow deploy = git pull | pip install | ./alisha.sh
                name_part, steps_part = arg.split("=", 1)
                name = name_part.strip()
                steps = [s.strip() for s in steps_part.split("|") if s.strip()]
                msg = memory.save_procedure(name=name, steps=steps)
                print(f"\n[Procedural] {msg}")
            else:
                matches = memory.search_procedures(arg)
                if matches:
                    p = matches[0]
                    print(f"\nProcedure '{p['name']}':")
                    for i, step in enumerate(p.get("steps", [])):
                        print(f"  {i+1}. {step}")
                else:
                    print(f"\n[Procedural] No procedure found matching '{arg}'.")
            continue

        # Model Versioning Commands
        elif lower_q in ("/version", "/versions", "/models"):
            version_mgr.print_version_table()
            continue

        elif lower_q == "/rollback":
            ok, msg = version_mgr.rollback()
            print(f"\n[Model Version] {msg}")
            if ok:
                new_adapter = version_mgr.get_active_adapter_path()
                load_active_model(new_adapter)
                print(f"[Model Version] Hot-swap complete: Now running {version_mgr.get_active_version_id()}!")
            continue

        elif lower_q in ("/fresh", "/base", "/reset"):
            ok, msg = version_mgr.rollback_to_fresh()
            print(f"\n[Model Version] {msg}")
            if ok:
                load_active_model(None)
                print("[Model Version] Hot-swap complete: Reset to Fresh Base Model (v0)!")
            continue

        elif lower_q.startswith("/switch "):
            target_vid = q[len("/switch ") :].strip()
            ok, msg = version_mgr.activate_version(target_vid)
            print(f"\n[Model Version] {msg}")
            if ok:
                new_adapter = version_mgr.get_active_adapter_path()
                load_active_model(new_adapter)
                print(f"[Model Version] Hot-swap complete: Switched to {target_vid}!")
            continue

        elif lower_q.startswith("/download"):
            target_mid = q[len("/download") :].strip()
            if not target_mid:
                print("Usage: /download <huggingface_model_id> (e.g. /download Qwen/Qwen2.5-1.5B-Instruct)")
                continue
            try:
                download_base_model(target_mid, verbose=True)
            except Exception as e:
                print(f"Error downloading: {e}")
            continue

        # Voice STT
        elif lower_q == "/speak":
            spoken_text = voice.record_and_transcribe(duration_seconds=5)
            if not spoken_text:
                print("[Voice] No speech detected.")
                continue
            if spoken_text.startswith("["):
                print(f"\n{spoken_text}")
                continue
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
            print("\n=== DYNAMICALLY DISCOVERED COGNITIVE MEMORY ===")
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
            print("  /briefing                Generate proactive executive system briefing")
            print("  /good                    Record positive feedback on previous answer (RLHF)")
            print("  /bad <better_answer>     Correct previous answer for continuous alignment")
            print("  /episodes | /diary       View recent daily milestones and activity timeline")
            print("  /workflow [name]         View or save developer execution procedures")
            print("  /version | /models       View all model versions and active status")
            print("  /rollback                Hot-swap back to previous model version")
            print("  /fresh                   Revert to un-finetuned Fresh Base Model (v0)")
            print("  /switch <vX>             Switch active model to version vX")
            print("  /download <model_id>     Download and cache a Hugging Face model")
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
            print("  /remember <key>: <val>   Save a custom fact to memory")
            print("  exit                     Quit")
            continue

        print("\nAlisa >", answer_question(q))


if __name__ == "__main__":
    main()
