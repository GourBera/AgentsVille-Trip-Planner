# AgentsVille Trip Planner — Python reference implementation

This repository contains a compact, runnable implementation of the "AgentsVille" trip-planning project. The original project is provided as a Jupyter notebook with a guided assignment that implements two agents, evaluation functions, and a ReAct-style revision loop.

Project goals
- Demonstrate advanced LLM reasoning patterns (role-based prompting, chain-of-thought, ReAct) applied to itinerary planning.
- Provide a runnable Python baseline with mocked APIs, evaluation hooks, and tools for deterministic or LLM-driven planning.

Repository layout
- [agentsville.py](agentsville.py#L1) — Main program: graph executor, deterministic planner, evaluator, tracer, and narration.
- [project_lib.py](project_lib.py#L1) — Mocked APIs, helper functions, activity/weather sample data, and utility helpers used by the planner and notebook.
- [project 1.ipynb](project%201.ipynb) — The notebook used for the Udacity assignment; contains the `ItineraryAgent`, `ItineraryRevisionAgent`, evaluation functions, and tools.
- [REPORT.md](REPORT.md#L1) — Short mapping of the implementation to the project rubric.
- `requirements.txt` — Python dependencies used by the notebook and scripts.

Notebook highlights
- `ItineraryAgent` — Generates an initial TravelPlan (day-by-day itinerary) from a `VacationInfo` object, using a Chain-of-Thought system prompt and a TravelPlan JSON output format.
- `ItineraryRevisionAgent` — A ReAct-based agent that iteratively revises the itinerary using tools (THOUGHT → ACTION → OBSERVATION cycles) and must run evaluation checks before returning the final plan.
- Tools defined in the notebook: `calculator_tool`, `get_activities_by_date_tool`, `run_evals_tool`, and `final_answer_tool`.
- Evaluation functions validate: dates, total cost accuracy, budget compliance, activity existence (no hallucinations), interest coverage per traveler, weather compatibility, and incorporation of traveler feedback (e.g., at least 2 activities per day).

Key features
- Deterministic planner + optional LLM-driven agents (notebook shows how to wire OpenAI/Vocareum endpoints).
- Reproducible evaluation pipeline: eval functions and a `get_eval_results` helper to collect failures.
- ReAct revision loop with explicit tool-contract rules (the notebook enforces a strict ordering: `run_evals_tool` -> `final_answer_tool`).

Quick start (notebook)
1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Launch the notebook (recommended) and run the cells in order:

```bash
jupyter notebook "project 1.ipynb"
```

3. Optional: configure an OpenAI-compatible client (the notebook includes an example `OpenAI` client construction for Vocareum). Set `OPENAI_API_KEY` or adjust the `client` construction in the notebook before running LLM-powered cells.

Notes on dependencies and models
- The notebook uses `pandas`, `pydantic`, `numexpr`, and optionally an OpenAI client. See `requirements.txt` for exact pins.
- The notebook defines a small `OpenAIModel` enum and sets a default model (`gpt-4.1-mini`) for interactive runs; you can change this to a different model in the notebook cells.

Running the script
- `agentsville.py` provides a deterministic runner that does not require LLMs; run with `python agentsville.py`.
- To exercise the full LLM-driven notebook flow and the ReAct revision loop, open and run [project 1.ipynb](project%201.ipynb).

Extending the project
- Replace the deterministic planner nodes with LLM-powered implementations or swap the executor for `langgraph` to run graph-based workflows.
- Add unit tests for the evaluation functions and the mocked APIs in `project_lib.py` to validate behavior.

Where to look next
- Start with [project 1.ipynb](project%201.ipynb) to see the assignment, system prompts, agents, and evaluation flow.
- Inspect [project_lib.py](project_lib.py#L1) for mocked endpoints used by the notebook (weather, activities, activity-by-id).
- Review [agentsville.py](agentsville.py#L1) if you want a non-notebook, script-based runner.

If you'd like, I can:
- Convert notebook agent cells into standalone Python modules, add unit tests for the evals, or wire a specific OpenAI model/credentials—tell me which and I will implement it.
