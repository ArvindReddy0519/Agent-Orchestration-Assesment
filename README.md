# Agent-Orchestration-Assesment

An Agentic Orchestration for full SDLC end to end automation controlled by human in loop governance.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Get a free API key from [Groq Console](https://console.groq.com/) and export it:

   ```bash
   export GROQ_API_KEY='your-key-here'
   ```

   Optional: override the default model (`llama-3.3-70b-versatile`):

   ```bash
   export GROQ_MODEL='llama-3.3-70b-versatile'
   ```

   You can also place `GROQ_API_KEY` in a `.env` file in the project root (loaded automatically).

## Run the human-in-the-loop demo

```bash
python scripts/run_hitl_demo.py --scenario approved
python scripts/run_hitl_demo.py --scenario rejected
```

The graph uses **LangGraph** with a **Groq**-backed LLM for planner, coder, and reviewer nodes. Plan approval still pauses on `interrupt()` until you resume with `Command(resume="Approved")` or `Command(resume="Rejected: ...")`.
