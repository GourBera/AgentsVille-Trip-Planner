"""AgentsVille Trip Planner (Python module)

This script reimplements the notebook-based project as a single Python file.
It uses a small graph-style execution pattern (LangGraph-compatible when
available) and records traces intended for LangSmith. If `langgraph` and
`langsmith` are installed, the script will try to use them; otherwise it
falls back to a local, dependency-free implementation so the project can
be run without extra packages.

Run:
    python agentsville.py

Environment:
        - Optionally set `OPENAI_API_KEY` to enable LLM-based evaluation and
            narration. If not provided, a deterministic fallback planner is used.
        - Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to enable
            automatic LangSmith tracing.
        - For APAC workspaces, set
            `LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com`.

Files created/used:
    - project_lib.py: helper functions and mocked data (already in workspace)

This file implements the project rubric steps: data validation, data
collection, itinerary generation, evaluation, revision, and narration.
"""

from __future__ import annotations

import json
import os
import datetime
import logging
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any, Optional, TypedDict, cast

from dataclasses import dataclass
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not available; skip loading .env and rely on environment
    pass

# Bootstrap key early so @traceable wrappers can authenticate before
# build_and_run() executes and calls configure_langsmith_from_env().
if os.environ.get("LANGSMITH_API_KEY") is None and os.environ.get("API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.environ["API_KEY"]

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False
    StateGraph = None
    START = "__start__"
    END = "__end__"

try:
    from langsmith import traceable
except Exception:
    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN_OPENAI = True
except Exception:
    HAS_LANGCHAIN_OPENAI = False
    ChatOpenAI = None

import urllib.request as _urlreq
import time as _time
import ssl as _ssl
from urllib import request as _urlreq2
import urllib.error as _urlerr
import ssl as _ssl

from project_lib import (
    Interest,
    call_weather_api_mocked,
    call_activities_api_mocked,
    INCLIMATE_WEATHER_CONDITIONS,
)
from decimal import Decimal

try:
    from langsmith.client import Client as LangSmithClient
except Exception:
    LangSmithClient = None

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_VACATION_INFO = {
    "travelers": [
        {"name": "Gour", "age": 32, "interests": ["hiking", "music", "comedy", "technology", "reading"]},
        {"name": "Babai", "age": 30, "interests": ["reading", "music", "theatre", "art", "sports"]},
    ],
    "destination": "AgentsVille",
    "date_of_arrival": "2025-06-10",
    "date_of_departure": "2025-06-13",
    "budget": 90,
    # Users can request how many activities per day they'd like (default 2)
    "activities_per_day": 1,
}


def _normalize_langsmith_endpoint(raw_endpoint: Optional[str]) -> Optional[str]:
    if not raw_endpoint:
        return None
    endpoint = raw_endpoint.strip().strip('"').strip("'")
    if not endpoint:
        return None
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = "https://" + endpoint.lstrip("/")

    # Users sometimes copy the UI org URL. Convert it to API endpoint.
    if "apac.smith.langchain.com/o/" in endpoint:
        endpoint = "https://apac.api.smith.langchain.com"
    elif "eu.smith.langchain.com/o/" in endpoint:
        endpoint = "https://eu.api.smith.langchain.com"
    elif "smith.langchain.com/o/" in endpoint and "apac" not in endpoint and "eu." not in endpoint:
        endpoint = "https://api.smith.langchain.com"

    return endpoint.rstrip("/")


def _apply_langsmith_project_url(raw_url: Optional[str]) -> Dict[str, Optional[str]]:
    """Extract endpoint/workspace/project from a LangSmith project URL.

    Example URL:
    https://apac.smith.langchain.com/o/<workspace-id>/projects/p/<project-id>?onboarding=AgentsVille
    """
    if not raw_url:
        return {"endpoint": None, "workspace_id": None, "project": None}

    url = raw_url.strip().strip('"').strip("'")
    if not url:
        return {"endpoint": None, "workspace_id": None, "project": None}

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path_parts = [p for p in parsed.path.split("/") if p]
    query = parse_qs(parsed.query)

    workspace_id = None
    if len(path_parts) >= 2 and path_parts[0] == "o":
        workspace_id = path_parts[1]

    endpoint = None
    if host == "apac.smith.langchain.com":
        endpoint = "https://apac.api.smith.langchain.com"
    elif host == "eu.smith.langchain.com":
        endpoint = "https://eu.api.smith.langchain.com"
    elif host.endswith("smith.langchain.com"):
        endpoint = "https://api.smith.langchain.com"

    project = None
    onboarding = query.get("onboarding", [])
    if onboarding and onboarding[0].strip():
        project = onboarding[0].strip()

    return {"endpoint": endpoint, "workspace_id": workspace_id, "project": project}


def configure_langsmith_from_env() -> Dict[str, Optional[str]]:
    """Prepare LangSmith env vars per trace-with-langgraph docs.

    - Mirrors API_KEY -> LANGSMITH_API_KEY for existing local setup.
    - Enables tracing by default for this script.
    - Prefers LANGSMITH_ENDPOINT and normalizes common copied UI URLs.
    """
    if os.environ.get("LANGSMITH_API_KEY") is None and os.environ.get("API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.environ["API_KEY"]

    if os.environ.get("LANGSMITH_TRACING") is None:
        os.environ["LANGSMITH_TRACING"] = "true"

    # If a specific project URL wasn't provided, default to the user's
    # supplied project so traces are recorded in the intended LangSmith
    # project. Users can override this by setting LANGSMITH_PROJECT_URL.
    project_url = os.environ.get("LANGSMITH_PROJECT_URL")
    if not project_url:
        os.environ["LANGSMITH_PROJECT_URL"] = (
            "https://apac.smith.langchain.com/o/160fb1d8-c541-4dc1-90d8-4938001bb691/projects/p/567b6371-9c62-4a71-9b6b-4debdd82aae8"
        )
        project_url = os.environ.get("LANGSMITH_PROJECT_URL")
    parsed_project_cfg = _apply_langsmith_project_url(project_url)

    if parsed_project_cfg.get("workspace_id"):
        os.environ["LANGSMITH_WORKSPACE_ID"] = cast(str, parsed_project_cfg["workspace_id"])

    if parsed_project_cfg.get("project"):
        os.environ["LANGSMITH_PROJECT"] = cast(str, parsed_project_cfg["project"])

    endpoint_candidate = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or parsed_project_cfg.get("endpoint")
        or os.environ.get("LANGSMITH_API_URL")
        or os.environ.get("LANGSMITH_BASE_URL")
        or os.environ.get("LANGSMITH_HOST")
    )
    endpoint = _normalize_langsmith_endpoint(endpoint_candidate)
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    return {
        "enabled": os.environ.get("LANGSMITH_TRACING"),
        "api_key_present": "yes" if bool(os.environ.get("LANGSMITH_API_KEY")) else "no",
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT"),
        "workspace_id": os.environ.get("LANGSMITH_WORKSPACE_ID"),
        "project": os.environ.get("LANGSMITH_PROJECT"),
    }


@dataclass
class Traveler:
    name: str
    age: int
    interests: List[Interest]


@dataclass
class VacationInfo:
    travelers: List[Traveler]
    destination: str
    date_of_arrival: datetime.date
    date_of_departure: datetime.date
    budget: int
    activities_per_day: int = 2

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VacationInfo":
        travelers = []
        for t in d.get("travelers", []):
            interests = [Interest(i) if not isinstance(i, Interest) else i for i in t.get("interests", [])]
            travelers.append(Traveler(name=t.get("name"), age=int(t.get("age")), interests=interests))
        doa = d.get("date_of_arrival")
        dod = d.get("date_of_departure")
        if isinstance(doa, str):
            doa = datetime.date.fromisoformat(doa)
        if isinstance(dod, str):
            dod = datetime.date.fromisoformat(dod)
        activities_per_day = int(d.get("activities_per_day", 2))
        return VacationInfo(
            travelers=travelers,
            destination=d.get("destination"),
            date_of_arrival=doa,
            date_of_departure=dod,
            budget=int(d.get("budget", 0)),
            activities_per_day=activities_per_day,
        )


@dataclass
class Weather:
    date: datetime.date
    temperature: float
    temperature_unit: str
    condition: str


@dataclass
class Activity:
    activity_id: str
    name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    location: str
    description: str
    price: int
    related_interests: List[Interest]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "name": self.name,
            "start_time": self.start_time.isoformat() if isinstance(self.start_time, datetime.datetime) else str(self.start_time),
            "end_time": self.end_time.isoformat() if isinstance(self.end_time, datetime.datetime) else str(self.end_time),
            "location": self.location,
            "description": self.description,
            "price": self.price,
            "related_interests": [i.value for i in self.related_interests],
        }


@dataclass
class ActivityRecommendation:
    activity: Activity
    reasons_for_recommendation: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"activity": self.activity.to_dict(), "reasons_for_recommendation": list(self.reasons_for_recommendation)}


@dataclass
class ItineraryDay:
    date: datetime.date
    weather: Weather
    activity_recommendations: List[ActivityRecommendation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "weather": {"temperature": self.weather.temperature, "temperature_unit": self.weather.temperature_unit, "condition": self.weather.condition},
            "activity_recommendations": [ar.to_dict() for ar in self.activity_recommendations],
        }


@dataclass
class TravelPlan:
    city: str
    start_date: datetime.date
    end_date: datetime.date
    total_cost: int
    itinerary_days: List[ItineraryDay]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_cost": self.total_cost,
            "itinerary_days": [d.to_dict() for d in self.itinerary_days],
        }


# Tracing is handled by LangSmith via the `traceable` decorator when
# available. We no longer keep a local tracer or write traces to disk.


class PlannerState(TypedDict, total=False):
    vacation_info: VacationInfo
    weather_for_dates: List[Dict[str, Any]]
    activities_for_dates: List[Dict[str, Any]]
    travel_plan: TravelPlan
    eval_results: Dict[str, Any]
    narration: str
    planner_unmet_requests: List[Dict[str, Any]]


def _build_narration_prompt(travel_plan: TravelPlan, planner_notes: List[Dict[str, Any]]) -> str:
    lines = [
        "Create a concise travel-plan narration.",
        f"City: {travel_plan.city}",
        f"Start date: {travel_plan.start_date}",
        f"End date: {travel_plan.end_date}",
        f"Total cost: {travel_plan.total_cost}",
        "Days:",
    ]
    for day in travel_plan.itinerary_days:
        lines.append(f"- {day.date} | weather={day.weather.condition} | activities={len(day.activity_recommendations)}")
        for ar in day.activity_recommendations:
            lines.append(f"  * {ar.activity.name} ({ar.activity.price})")

    if planner_notes:
        lines.append("Planner notes:")
        for note in planner_notes:
            lines.append(
                f"- requested={note['requested']} on {note['date']}, selected={note['selected']}, attempts={note.get('attempts', 0)}"
            )

    lines.append("Return plain text only.")
    return "\n".join(lines)


def _call_local_ollama(prompt: str) -> Dict[str, Any]:
    """Call a local Ollama instance using the LOCAL_LLM_URL and model.

    Returns a dict: {text, tokens_estimate, cost_estimate, first_token_ms}
    """
    url = os.environ.get("LOCAL_LLM_URL") or os.environ.get("LOCAL_LLM_ENDPOINT") or "http://localhost:11434/api/generate"
    model = os.environ.get("LOCAL_LLM_MODEL") or "llama2"
    cost_per_1k = float(os.environ.get("LOCAL_LLM_COST_PER_1K", "0"))

    body = json.dumps({"model": model, "prompt": prompt}).encode("utf-8")
    req = _urlreq.Request(url, data=body, headers={"Content-Type": "application/json"})
    # Allow self-signed/local certs
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    start = _time.time()
    try:
        with _urlreq.urlopen(req, timeout=30, context=ctx) as resp:
            # try to read first byte to measure first-token latency
            try:
                # read a small chunk for first-token timing
                first_chunk = resp.read(64)
                first_token_ms = int((_time.time() - start) * 1000)
                # then read remainder
                rest = resp.read()
                raw = (first_chunk or b"") + (rest or b"")
            except Exception:
                raw = resp.read()
                first_token_ms = int((_time.time() - start) * 1000)

            text = raw.decode("utf-8", errors="ignore")
            # Try to parse JSON if Ollama returned structured JSON
            try:
                j = json.loads(text)
                # Common Ollama shape: {'model':..., 'generated': 'text' } or similar
                if isinstance(j, dict):
                    if j.get("generated"):
                        text = j.get("generated")
                    elif j.get("output"):
                        text = j.get("output")
            except Exception:
                pass

            # crude token estimate: 1.3 tokens per word
            tokens_est = int(max(1, len(prompt.split()) * 1.3))
            cost = tokens_est / 1000.0 * cost_per_1k
            return {"text": text, "tokens": tokens_est, "cost": cost, "first_token_ms": first_token_ms, "model": model, "provider": "ollama",}
    except Exception as e:
        return {"error": str(e)}


def _call_openai_rest(prompt: str) -> Dict[str, Any]:
    """Call OpenAI-compatible REST endpoint and return text, tokens, cost, first_token_ms."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY missing"}
    base = os.environ.get("OPENAI_API_BASE") or "https://api.openai.com"
    model = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    cost_per_1k = float(os.environ.get("OPENAI_COST_PER_1K", "0"))

    url = base.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode("utf-8")
    req = _urlreq.Request(url, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    start = _time.time()
    try:
        with _urlreq.urlopen(req, timeout=30) as resp:
            # measure first token latency by reading small chunk
            try:
                first_chunk = resp.read(64)
                first_token_ms = int((_time.time() - start) * 1000)
                rest = resp.read()
                raw = (first_chunk or b"") + (rest or b"")
            except Exception:
                raw = resp.read()
                first_token_ms = int((_time.time() - start) * 1000)

            text = raw.decode("utf-8", errors="ignore")
            try:
                j = json.loads(text)
            except Exception:
                return {"error": "invalid json from openai", "raw": text}

            # extract generated content
            try:
                choices = j.get("choices", [])
                if choices:
                    msg = choices[0].get("message") or choices[0].get("delta") or {}
                    generated = msg.get("content") or choices[0].get("text") or ""
                else:
                    generated = ""
            except Exception:
                generated = ""

            usage = j.get("usage") or {}
            total_tokens = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            try:
                total_tokens = int(total_tokens)
            except Exception:
                total_tokens = 0

            cost = total_tokens / 1000.0 * cost_per_1k
            return {"text": generated, "tokens": total_tokens, "cost": cost, "first_token_ms": first_token_ms, "model": model, "provider": "openai", "raw": j}
    except Exception as e:
        return {"error": str(e)}


@traceable(run_type="chain", name="collect_data")
def node_collect_data(ctx: Dict[str, Any]):
    vacation_info: VacationInfo = ctx["vacation_info"]

    # Gather weather for each date
    weather_for_dates = []
    current = vacation_info.date_of_arrival
    while current <= vacation_info.date_of_departure:
        wf = call_weather_api_mocked(date=current.strftime("%Y-%m-%d"), city=vacation_info.destination)
        if wf:
            weather_for_dates.append(wf)
        current += datetime.timedelta(days=1)

    # Gather activities for each date
    activities_for_dates = []
    current = vacation_info.date_of_arrival
    while current <= vacation_info.date_of_departure:
        acts = call_activities_api_mocked(date=current.strftime("%Y-%m-%d"), city=vacation_info.destination)
        activities_for_dates.extend(acts)
        current += datetime.timedelta(days=1)

    ctx["weather_for_dates"] = weather_for_dates
    ctx["activities_for_dates"] = activities_for_dates


def _parse_activity_dict(act: Dict[str, Any]) -> Activity:
    # Convert ISO datetime-like strings to datetime objects
    start = act.get("start_time")
    end = act.get("end_time")
    if isinstance(start, str):
        start = datetime.datetime.fromisoformat(start.replace(" ", "T"))
    if isinstance(end, str):
        end = datetime.datetime.fromisoformat(end.replace(" ", "T"))

    related = [Interest(i) for i in act.get("related_interests", [])]
    return Activity(
        activity_id=act.get("activity_id"),
        name=act.get("name"),
        start_time=start,
        end_time=end,
        location=act.get("location"),
        description=act.get("description"),
        price=int(act.get("price", 0)),
        related_interests=related,
    )


@traceable(run_type="chain", name="plan_itinerary")
def node_plan_itinerary(ctx: Dict[str, Any]):
    """Create a deterministic itinerary without calling an LLM.

    This node implements the core rubric: choose activities by interest,
    avoid outdoor-only activities on inclement weather, ensure at least 2
    activities per day, and compute the total cost using exact sums.
    """
    vacation_info: VacationInfo = ctx["vacation_info"]
    activities_for_dates = ctx.get("activities_for_dates", [])

    itinerary_days: List[ItineraryDay] = []
    current = vacation_info.date_of_arrival
    while current <= vacation_info.date_of_departure:
        date_str = current.strftime("%Y-%m-%d")
        wf = call_weather_api_mocked(date=date_str, city=vacation_info.destination)
        weather_obj = Weather(
            date=current,
            temperature=wf.get("temperature", 0),
            temperature_unit=wf.get("temperature_unit", "celsius"),
            condition=wf.get("condition", "unknown"),
        )

        # Find activities for this date
        acts_raw = call_activities_api_mocked(date=date_str, city=vacation_info.destination)
        parsed_acts = []
        for ar in acts_raw:
            try:
                parsed_acts.append(_parse_activity_dict(ar))
            except Exception:
                continue

        # First, prefer activities matching travelers' interests
        all_interests = set()
        for t in vacation_info.travelers:
            all_interests.update([i.value if isinstance(i, Interest) else str(i) for i in t.interests])

        # Filter out clearly incompatible outdoor activities when inclement
        inclement = weather_obj.condition in INCLIMATE_WEATHER_CONDITIONS

        # Score activities by interest overlap and indoor safety
        def score_activity(a: Activity) -> int:
            interest_score = len(set([i.value for i in a.related_interests]) & set(all_interests))
            indoor_bonus = 1 if "indoor" in a.description.lower() or "inside" in a.description.lower() or "covered" in a.description.lower() else 0
            return interest_score * 10 + indoor_bonus

        candidates = sorted(parsed_acts, key=lambda a: (-score_activity(a), a.price))

        # Allow users to request `activities_per_day`. Try multiple attempts
        # to meet the request; if after retries we still cannot meet it,
        # record a planner-level notification.
        requested = getattr(vacation_info, "activities_per_day", 2)
        max_attempts = 3
        attempts = 0
        picks: List[Activity] = []

        extras = [ _parse_activity_dict(a) for a in activities_for_dates ]
        extras_sorted = sorted(extras, key=lambda a: (a.price, -score_activity(a)))

        while attempts < max_attempts and len(picks) < requested:
            # On first attempt respect inclement filtering; on retries relax it
            inclement_check = inclement and attempts == 0

            # Build filtered list according to current inclement policy
            curr_filtered = []
            for a in candidates:
                desc = a.description.lower()
                if inclement_check and ("outdoor" in desc or "open-air" in desc or "outdoor" in a.name.lower()):
                    if "indoor" in desc or "covered" in desc or "inside" in desc:
                        curr_filtered.append(a)
                    else:
                        continue
                else:
                    curr_filtered.append(a)

            # pick top matching activities (ordered by score then price)
            picks = curr_filtered[:requested]

            # fill from other available activities if still short
            i = 0
            while len(picks) < requested and i < len(extras_sorted):
                candidate = extras_sorted[i]
                if candidate.activity_id not in [p.activity_id for p in picks]:
                    picks.append(candidate)
                i += 1

            if len(picks) >= requested:
                break
            attempts += 1

        if len(picks) < requested:
            # record planner-level unmet request so evaluator and narration can notify the user
            ctx.setdefault("planner_unmet_requests", []).append({"date": date_str, "requested": requested, "selected": len(picks), "attempts": attempts})

        reasons = [f"Matches interests or is low-cost option" for _ in picks]
        recs = [ActivityRecommendation(activity=picks[i], reasons_for_recommendation=[reasons[i]]) for i in range(len(picks))]

        itinerary_days.append(ItineraryDay(date=current, weather=weather_obj, activity_recommendations=recs))
        current += datetime.timedelta(days=1)

    total_cost = sum(ar.activity.price for day in itinerary_days for ar in day.activity_recommendations)

    travel_plan = TravelPlan(
        city=vacation_info.destination,
        start_date=vacation_info.date_of_arrival,
        end_date=vacation_info.date_of_departure,
        total_cost=total_cost,
        itinerary_days=itinerary_days,
    )

    ctx["travel_plan"] = travel_plan


@traceable(run_type="chain", name="run_evals")
def node_run_evals(ctx: Dict[str, Any]):
    """Run deterministic evaluation functions mirroring the notebook's checks."""
    vacation_info: VacationInfo = ctx["vacation_info"]
    travel_plan: TravelPlan = ctx.get("travel_plan")

    failures = []

    # 1) start/end date match
    if vacation_info.date_of_arrival != travel_plan.start_date or vacation_info.date_of_departure != travel_plan.end_date:
        failures.append("Dates do not match arrival/departure")

    # 2) total cost accurate
    calc_total = sum(ar.activity.price for day in travel_plan.itinerary_days for ar in day.activity_recommendations)
    if calc_total != travel_plan.total_cost:
        failures.append(f"Total cost mismatch: calculated {calc_total} vs stated {travel_plan.total_cost}")

    # 3) within budget
    if travel_plan.total_cost > vacation_info.budget:
        failures.append(f"Total cost exceeds budget: {travel_plan.total_cost} > {vacation_info.budget}")

    # 4) each traveler has at least one matching activity
    traveler_hits = {t.name: 0 for t in vacation_info.travelers}
    for t in vacation_info.travelers:
        for day in travel_plan.itinerary_days:
            for ar in day.activity_recommendations:
                if set([i.value for i in t.interests]) & set([ri.value for ri in ar.activity.related_interests]):
                    traveler_hits[t.name] += 1
    missing = [name for name, cnt in traveler_hits.items() if cnt == 0]
    if missing:
        failures.append(f"Travelers with no matching activities: {missing}")

    # 5) weather compatibility (simple deterministic check)
    incompatible = []
    for day in travel_plan.itinerary_days:
        if day.weather.condition in INCLIMATE_WEATHER_CONDITIONS:
            for ar in day.activity_recommendations:
                desc = ar.activity.description.lower()
                if not ("indoor" in desc or "inside" in desc or "covered" in desc or "museum" in desc or "auditorium" in desc or "studio" in desc):
                    incompatible.append(ar.activity.activity_id)
    if incompatible:
        failures.append(f"Activities incompatible with weather: {incompatible}")

    # 6) per-day activity count constraints: ensure matches requested activities_per_day (flag if not met)
    requested = getattr(vacation_info, "activities_per_day", 2)
    for day in travel_plan.itinerary_days:
        n = len(day.activity_recommendations)
        if n < requested:
            failures.append(f"Less than {requested} activities on {day.date}: {n}")
        if n > requested:
            failures.append(f"More than {requested} activities on {day.date}: {n}")

    # Include planner notifications (unmet requests after retries)
    for note in ctx.get("planner_unmet_requests", []):
        failures.append(f"Planner couldn't meet requested {note['requested']} activities on {note['date']}: selected {note['selected']} after {note.get('attempts', 0)} attempts")

    ctx["eval_results"] = {"success": len(failures) == 0, "failures": failures}


@traceable(run_type="chain", name="narrate")
def node_narrate(ctx: Dict[str, Any]):
    travel_plan: TravelPlan = ctx.get("travel_plan")

    # To populate token/cost/first-token fields in LangSmith, we try an
    # actual LLM run. Prefer local endpoints, then OpenAI REST, then
    # the langchain wrapper, and finally a deterministic fallback.
    prompt = _build_narration_prompt(travel_plan, ctx.get("planner_unmet_requests", []))

    # 1) Local provider: Ollama or a generic local LLM HTTP endpoint
    local_provider = os.environ.get("LOCAL_LLM_PROVIDER") or os.environ.get("LOCAL_LLM")
    local_url = os.environ.get("LOCAL_LLM_URL") or os.environ.get("LLM_LOCAL_URL")
    if local_provider and local_provider.lower() == "ollama":
        try:
            start_time = datetime.datetime.now(datetime.timezone.utc)
            res = _call_local_ollama(prompt)
            end_time = datetime.datetime.now(datetime.timezone.utc)
            if not res.get("error"):
                narration_text = res.get("text")
                # record LLM usage to LangSmith if client available
                try:
                    tokens = int(res.get("tokens", 0) or 0)
                    cost = float(res.get("cost", 0.0) or 0.0)
                    first_token_ms = int(res.get("first_token_ms", 0) or 0)
                    first_token_time = start_time + datetime.timedelta(milliseconds=first_token_ms)
                    if LangSmithClient is not None:
                        api_url = os.environ.get("LANGSMITH_ENDPOINT")
                        api_key = os.environ.get("LANGSMITH_API_KEY")
                        project_name = os.environ.get("LANGSMITH_PROJECT")
                        client = LangSmithClient(api_url=api_url, api_key=api_key, workspace_id=os.environ.get("LANGSMITH_WORKSPACE_ID"))
                        try:
                            # Try to find a recent run created by the traceable decorator
                            # and update it with token/cost/first_token metrics. If none
                            # is found, fall back to creating a new child run.
                            updated = False
                            try:
                                recent = list(client.list_runs(limit=50))
                                # choose candidate by name or by inputs containing the prompt
                                candidates = []
                                for rr in recent:
                                    name = getattr(rr, "name", "") or ""
                                    inputs = getattr(rr, "inputs", {}) or {}
                                    start = getattr(rr, "start_time", None)
                                    candidates.append((rr, name, inputs, start))

                                def score_candidate(c):
                                    rr, name, inputs, start = c
                                    s = 0
                                    if "narrat" in name.lower() or "narration" in name.lower():
                                        s += 10
                                    # exact prompt match boosts score
                                    if isinstance(inputs, dict) and any(prompt.strip() in str(v) for v in inputs.values()):
                                        s += 50
                                    # recent start_time preferred
                                    return s

                                candidates_sorted = sorted(candidates, key=score_candidate, reverse=True)
                                if candidates_sorted and score_candidate(candidates_sorted[0]) > 0:
                                    run_obj = candidates_sorted[0][0]
                                    run_id = getattr(run_obj, "id", None)
                                    if run_id:
                                        client.update_run(
                                            run_id,
                                            completion_tokens=tokens,
                                            total_tokens=tokens,
                                            completion_cost=Decimal(str(cost)),
                                            total_cost=Decimal(str(cost)),
                                            first_token_time=first_token_time,
                                            api_key=api_key,
                                            api_url=api_url,
                                        )
                                        updated = True
                            except Exception as inner_e:
                                logger.info("Could not search/update recent runs: %s", inner_e)

                            if not updated:
                                client.create_run(
                                    name="agentsville_narration_llm",
                                    inputs={"prompt": prompt},
                                    outputs={"text": narration_text},
                                    run_type="llm",
                                    project_name=project_name,
                                    api_key=api_key,
                                    api_url=api_url,
                                    start_time=start_time,
                                    end_time=end_time,
                                    prompt_tokens=0,
                                    completion_tokens=tokens,
                                    total_tokens=tokens,
                                    first_token_time=first_token_time,
                                    total_cost=Decimal(str(cost)),
                                )
                        except Exception as e:
                            logger.warning("LangSmith create_run failed: %s", e)
                except Exception as e:
                    logger.warning("Failed to parse local LLM usage: %s", e)
                ctx["narration"] = narration_text
                return
        except Exception:
            pass
    if local_url:
        try:
            data = json.dumps({"prompt": prompt}).encode("utf-8")
            req = _urlreq.Request(local_url, data=data, headers={"Content-Type": "application/json"})
            ctxssl = _ssl.create_default_context()
            ctxssl.check_hostname = False
            ctxssl.verify_mode = _ssl.CERT_NONE
            start = _time.time()
            with _urlreq.urlopen(req, timeout=30, context=ctxssl) as resp:
                body = resp.read().decode("utf-8")
                try:
                    js = json.loads(body)
                    narration_text = js.get("text") or js.get("generated_text") or js.get("result") or js.get("output") or str(js)
                except Exception:
                    narration_text = body
                ctx["narration"] = narration_text
                return
        except Exception:
            pass

    # 2) OpenAI REST (preferred) if API key present
    if os.environ.get("OPENAI_API_KEY"):
        try:
            res = _call_openai_rest(prompt)
            if not res.get("error"):
                ctx["narration"] = res.get("text")
                return
        except Exception:
            pass

    # 3) LangChain OpenAI wrapper as a fallback
    if HAS_LANGCHAIN_OPENAI and ChatOpenAI is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            llm_cls = cast(Any, ChatOpenAI)
            llm = llm_cls(model=model_name, temperature=0)
            response = llm.invoke(prompt)
            narration_text = response.content if isinstance(response.content, str) else str(response.content)
            ctx["narration"] = narration_text
            return
        except Exception:
            pass

    # Deterministic fallback narration
    lines = [f"Trip to {travel_plan.city} from {travel_plan.start_date} to {travel_plan.end_date}", f"Total cost: {travel_plan.total_cost}", "Itinerary:"]
    for day in travel_plan.itinerary_days:
        lines.append(f"- {day.date}: {day.weather.condition} - {len(day.activity_recommendations)} activities")
        for ar in day.activity_recommendations:
            lines.append(f"    * {ar.activity.name} ({ar.activity.price}) - {', '.join([i.value for i in ar.activity.related_interests])}")

    # Surface planner notifications (e.g., unmet activity-per-day requests)
    if ctx.get("planner_unmet_requests"):
        lines.append("Notifications:")
        for note in ctx.get("planner_unmet_requests", []):
            lines.append(f"- Could not meet requested {note['requested']} activities on {note['date']}: selected {note['selected']} after {note.get('attempts', 0)} attempts")

    narration = "\n".join(lines)
    ctx["narration"] = narration


def _run_with_langgraph(initial_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not HAS_LANGGRAPH or StateGraph is None:
        raise RuntimeError("LangGraph runtime requested but langgraph is not installed")

    def _collect(state: PlannerState) -> PlannerState:
        ctx = dict(state)
        node_collect_data(ctx)
        return cast(PlannerState, ctx)

    def _plan(state: PlannerState) -> PlannerState:
        ctx = dict(state)
        node_plan_itinerary(ctx)
        return cast(PlannerState, ctx)

    def _eval(state: PlannerState) -> PlannerState:
        ctx = dict(state)
        node_run_evals(ctx)
        return cast(PlannerState, ctx)

    def _narrate(state: PlannerState) -> PlannerState:
        ctx = dict(state)
        node_narrate(ctx)
        return cast(PlannerState, ctx)

    graph = StateGraph(PlannerState)
    graph.add_node("collect_data", _collect)
    graph.add_node("plan_itinerary", _plan)
    graph.add_node("run_evals", _eval)
    graph.add_node("narrate", _narrate)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "plan_itinerary")
    graph.add_edge("plan_itinerary", "run_evals")
    graph.add_edge("run_evals", "narrate")
    graph.add_edge("narrate", END)

    app = graph.compile()
    return app.invoke(initial_ctx)


def _run_with_lite_graph(initial_ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Lightweight sequential runner when langgraph isn't installed.
    ctx = dict(initial_ctx)
    node_collect_data(ctx)
    node_plan_itinerary(ctx)
    node_run_evals(ctx)
    node_narrate(ctx)
    return ctx


@traceable(run_type="chain", name="agentsville_trip_planner")
def build_and_run(vacation_input: dict[str, Any] | None = None) -> Dict[str, Any]:
    ls_config = configure_langsmith_from_env()

    # Validate input
    if vacation_input is None:
        vacation_input = DEFAULT_VACATION_INFO
    # convert date strings to date objects
    vi = vacation_input.copy()
    vi["date_of_arrival"] = datetime.date.fromisoformat(vi["date_of_arrival"]) if isinstance(vi["date_of_arrival"], str) else vi["date_of_arrival"]
    vi["date_of_departure"] = datetime.date.fromisoformat(vi["date_of_departure"]) if isinstance(vi["date_of_departure"], str) else vi["date_of_departure"]

    vacation_info = VacationInfo.from_dict(vi)

    ctx: Dict[str, Any] = {"vacation_info": vacation_info}

    if HAS_LANGGRAPH:
        result_ctx = _run_with_langgraph(ctx)
    else:
        result_ctx = _run_with_lite_graph(ctx)

    print(
        "LangSmith tracing config:",
        {
            "LANGSMITH_TRACING": ls_config.get("enabled"),
            "LANGSMITH_API_KEY": ls_config.get("api_key_present"),
            "LANGSMITH_ENDPOINT": ls_config.get("endpoint") or "default-us",
            "LANGSMITH_WORKSPACE_ID": ls_config.get("workspace_id") or "unset",
            "LANGSMITH_PROJECT": ls_config.get("project") or "unset",
        },
    )

    return result_ctx


def main():
    ctx = build_and_run()
    travel_plan: TravelPlan = ctx["travel_plan"]
    print("\n===== Final Travel Plan (summary) =====\n")
    print(json.dumps(travel_plan.to_dict(), indent=2))
    print("\n===== Narration =====\n")
    print(ctx["narration"])


if __name__ == "__main__":
    main()
