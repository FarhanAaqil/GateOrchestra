"""
assistant
=========
GateOrchestra as a chat assistant.

A cheap probe answers each message a few times; the gate reads how much the
samples agree and decides whether one model call is enough (STOP) or the
question needs the tool-using agent (ESCALATE). Answers stream back as
Markdown. Providers (Groq, Gemini, a local Ollama) sit behind one gateway with
key rotation and fallback.

The research benchmark (agents/, gate/, scripts/final_benchmark.py) is
separate and unchanged.
"""
