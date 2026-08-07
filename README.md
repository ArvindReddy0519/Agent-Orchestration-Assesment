# Agentic Orchestration Framework & Multi-Agent SDLC

An enterprise-grade, stateful multi-agent orchestration platform built with **LangGraph** and **Groq (LLaMA 3)**. The system automates the Software Development Life Cycle (SDLC)—from natural-language requirements to structured execution planning, incremental code generation, per-artifact review, bounded retry loops, project assembly, and human-in-the-loop (HITL) governance.

The framework generates complete software projects **dynamically from natural-language requirements**. Provide any product description—such as a **Payment Gateway**, **Event-Driven System**, **Inventory Management** service, or **URL Shortener**—and the orchestrator produces a decomposed task plan and incrementally builds a Java 17 / Spring Boot 3 codebase under `generated_project/`.

---

## Architecture & Multi-Agent Graph

The framework operates as a **stateful LangGraph** with checkpointing, enforcing governance checkpoints, dependency-aware scheduling, and self-correction cycles:

```text
[Requirement Input]
       │
       ▼
┌───────────────┐
│ Planner Node  │ ──► Structured JSON execution plan (architecture decisions + task graph)
└───────────────┘
       │
       ▼ (HITL Checkpoint: interrupt())
┌───────────────┐
│ Approval Node │ ──► Human-in-the-Loop review (Approved / Rejected + feedback)
└───────────────┘
       │
       ├─────────────────────────┐ (If Rejected)
       │ (If Approved)           ▼
       │                  ┌───────────────┐
       │                  │ Planner Loop  │ (Partial replan from human feedback)
       │                  └───────────────┘
       ▼
┌───────────────────────┐
│ Task Scheduler Node   │ ──► Selects ready tasks whose dependencies are satisfied
└───────────────────────┘
       │
       ├──────────────────────────────┐ (All tasks complete)
       │ (Next ready task)            ▼
       ▼                       ┌──────────────────┐
┌───────────────┐             │ Project Assembly │ ──► Merges artifacts into unified codebase
│ Coder Node    │             └──────────────────┘
└───────────────┘                       │
       │                                ▼
       ▼                       ┌──────────────────────┐
┌───────────────┐             │ Engineering Summary  │ ──► Deterministic aggregation of reviews
│ Reviewer Node │             └──────────────────────┘
└───────────────┘                       │
       │                                ├── (Pass) ──► [Generated Project]
       └──────── loop ──────────────────┤
                                        └── (Findings, retries remaining)
                                                  │
                                                  ▼
                                        ┌──────────────────────┐
                                        │ Task Invalidation Node │ ──► Clears affected tasks & reviews
                                        └──────────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────────┐
                                        │ Increment Retry Node   │
                                        └──────────────────────┘
                                                  │
                                                  └──► Task Scheduler (re-execute invalidated work)
```

**Per-task loop:** After each incremental coder run, the **Per-Artifact Reviewer** audits every file produced by that task individually (one LLM call per file). The scheduler then dispatches the next ready task until the full task graph is complete.

**Post-assembly:** Once all tasks finish, **Project Assembly** consolidates file artifacts, **Engineering Summary** aggregates per-artifact review results without an additional LLM call, and the runner persists output to `generated_project/`. Failed summaries trigger **Task Invalidation** and a bounded retry loop (up to three attempts) before the graph exits.

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Dynamic requirement-driven planning** | The planner accepts arbitrary natural-language requirements and emits a structured JSON execution plan rather than hardcoded scenario templates. |
| **Task-based orchestration** | Work is decomposed into 8–15 focused tasks (POM, entities, services, controllers, tests, etc.) with explicit IDs and dependency edges. |
| **Dependency-aware scheduling** | The scheduler dispatches only tasks whose dependencies are satisfied, preserving build order and enabling incremental context. |
| **Per-artifact code review** | Each generated file is reviewed in isolation immediately after its parent task completes, keeping reviewer prompts within model token limits. |
| **Deterministic review aggregation** | The engineering summary is produced in Python by classifying and counting approved artifacts, warnings, and failures—no LLM summarization step. |
| **Shared LangGraph state** | `completed_tasks`, `artifacts`, and `artifact_reviews` use reducers (`merge_dicts`) for safe concurrent updates; scalar fields are written by single nodes per step. |
| **Retry and task invalidation** | Review findings map to task IDs and file paths; downstream dependents are invalidated and regenerated on retry with scoped, bounded context. |
| **Human-in-the-Loop governance** | Plan approval pauses the graph via `interrupt()` before any code generation begins, ensuring architectural oversight. |

---

## Why Python and LangGraph?

**Python** provides a mature ecosystem for LLM orchestration (LangChain, Pydantic, Groq SDK) and rapid iteration on agent prompts, state schemas, and workflow logic—without requiring a separate runtime for the orchestrator itself.

**LangGraph** was chosen because SDLC automation is inherently **stateful and cyclical**: plan → approve → schedule → code → review → retry. LangGraph offers:

- A explicit **graph model** with conditional routing and checkpointing
- **`interrupt()` / `Command(resume=...)`** for production-grade HITL pauses
- **`Send`-based fan-out** for task dispatch (currently serialized to one task per wave for Groq TPM limits)
- **Annotated state reducers** for merging parallel artifact updates safely

Together, Python and LangGraph implement a governable agentic pipeline suitable for enterprise assessment and extension to persistent checkpoint stores (e.g., PostgreSQL) in production deployments.

---

## Design Trade-offs

The architecture deliberately moved away from **monolithic single-shot generation** toward **incremental task execution**:

| Monolithic approach | Incremental task graph |
|---------------------|------------------------|
| Entire codebase in one LLM call | One focused deliverable per coder invocation |
| Reviewer receives full project (token limit failures) | Per-artifact reviewer stays within context windows |
| All-or-nothing retry on any defect | Targeted invalidation of affected tasks and dependents |
| Planner output as prose | Structured JSON plan validated by Pydantic models |

**Trade-offs accepted:**

- **Higher LLM call count** — More planner, coder, and reviewer invocations per run, mitigated by one-task-per-wave scheduling and rate-limit retry logic.
- **Serialized task dispatch** — Parallel fan-out is disabled to respect Groq free-tier TPM limits; the graph structure supports re-enabling parallelism when capacity allows.
- **Heuristic task-to-error mapping** — Invalidation uses path and keyword matching rather than a formal traceability matrix; sufficient for bounded retry loops but not a substitute for full requirements traceability tooling.

These trade-offs prioritize **reliability, observability, and token-budget compliance** over minimum latency—appropriate for a production-oriented agentic SDLC platform.

---

## Project Directory Structure

```text
Agent-Orchestration-Assesment/
├── orchestrator/
│   ├── graph.py         # LangGraph state machine: nodes, edges, checkpointing
│   ├── nodes.py         # Planner, approval, scheduler, coder, reviewer, assembly, summary, invalidation
│   ├── state.py         # OrchestratorState TypedDict and reducer-annotated fields
│   ├── tasks.py         # Task / ExecutionPlan models, scheduling helpers, merge_dicts, path matching
│   ├── llm.py           # Groq client factory, rate-limit retry, deterministic review aggregation
│   └── file_writer.py   # Parses ### File: blocks and writes artifacts to disk
├── scripts/
│   ├── run_scenarios.py # Interactive requirement-driven runner with HITL plan approval
│   └── run_hitl_demo.py # Automated HITL regression demos (Approved / Rejected flows)
├── generated_project/   # Output directory for generated Java / Spring Boot artifacts
├── .env                 # Groq API key and optional model override (not committed)
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
└── README.md
```

### Module responsibilities

- **`graph.py`** — Wires the full workflow: planner → approval → scheduler ⇄ coder → reviewer → assembly → engineering summary → invalidation/retry.
- **`nodes.py`** — Agent node implementations, routing functions, and bounded-context helpers for coder retries.
- **`state.py`** — Shared graph state including `tasks`, `completed_tasks`, `artifacts`, `artifact_reviews`, `review_summary`, and `errors`.
- **`tasks.py`** — Pydantic `Task` / `ExecutionPlan` models, ready-task selection, downstream expansion for invalidation, and `merge_dicts` for LangGraph reducers.
- **`llm.py`** — Groq connectivity validation, cached `ChatGroq` instance, exponential backoff on 429 rate limits, and `aggregate_artifact_reviews()` for deterministic summaries.
- **`file_writer.py`** — Extracts `### File: <path>` markdown blocks from assembled code and writes files under `generated_project/`.

---

## Prerequisites

- Python 3.10+ with pip and virtual environment support
- Java 17 Development Kit (JDK 17) for compiling and testing generated code
- Apache Maven for dependency management and build verification
- A Groq API key (LLaMA 3 models supported via `GROQ_MODEL`)

---

## Setup

1. Create a virtual environment and install Python dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment variables — create a `.env` file in the root directory:

   ```bash
   cp .env.example .env
   ```

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

   Get a free key at [Groq Console](https://console.groq.com/keys). Supported models include `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`.

---

## Running the Framework

### 1. Interactive requirement-driven run

Run the end-to-end CLI with a natural-language requirement. The graph pauses at plan approval for your decision before code generation begins.

Pass a requirement on the command line:

```bash
python scripts/run_scenarios.py --requirement "Design a payment gateway with Stripe integration and JWT auth"
```

Or launch interactively and enter a multi-line requirement when prompted:

```bash
python scripts/run_scenarios.py
```

At the HITL checkpoint, respond with:

- `Approved` — proceed to task scheduling and incremental code generation
- `Rejected: <feedback>` — return to the planner with your architectural feedback

On completion, the runner prints the final state summary, assembled code, and file paths written under `generated_project/`.

### 2. Human-in-the-Loop (HITL) approved flow demo

Automated regression test simulating human plan approval without interactive input:

```bash
python scripts/run_hitl_demo.py --scenario approved
```

### 3. Human-in-the-Loop (HITL) rejection and feedback loop demo

Automated regression test where the plan is rejected and the planner replans from feedback:

```bash
python scripts/run_hitl_demo.py --scenario rejected
```

---

## Verifying the Generated Codebase

The runner writes generated artifacts into `generated_project/` after a successful graph run. To compile and execute the test suite (JUnit 5 + Mockito via Maven):

```bash
cd generated_project
mvn clean test
```

To run the generated Spring Boot application locally:

```bash
mvn spring-boot:run
```

Generated project structure, dependencies, and entry points vary by requirement—the orchestrator does not assume a fixed domain template.
