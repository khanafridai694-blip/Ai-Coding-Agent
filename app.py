"""SentinelCoder: a three-agent crew that writes, audits and hardens code.

Pipeline (CrewAI sequential process, Groq inference, Firestore history):
    1. Software Engineer ............... writes the first implementation
    2. Application Security Specialist . audits it for vulnerabilities
    3. QA and Refactoring Engineer ..... fixes the findings, refactors, writes tests

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# CrewAI reads these switches when it is imported. They turn off anonymous
# telemetry and the interactive "view your traces?" prompt that would otherwise
# block a Streamlit server for up to 20 seconds after every run. setdefault keeps
# any value you exported yourself.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st  # noqa: E402

try:
    from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402

    CREWAI_IMPORT_ERROR: Optional[str] = None
except Exception as import_exc:  # noqa: BLE001
    LLM = Agent = Crew = Process = Task = None  # type: ignore[assignment,misc]
    CREWAI_IMPORT_ERROR = f"{type(import_exc).__name__}: {import_exc}"

try:
    import firebase_admin  # noqa: E402
    from firebase_admin import credentials, firestore  # noqa: E402

    FIREBASE_IMPORT_ERROR: Optional[str] = None
except Exception as import_exc:  # noqa: BLE001
    firebase_admin = credentials = firestore = None  # type: ignore[assignment]
    FIREBASE_IMPORT_ERROR = f"{type(import_exc).__name__}: {import_exc}"

# st.set_page_config must be the first Streamlit command that draws anything.
st.set_page_config(page_title="SentinelCoder", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Groq model ID. Groq retired llama-3.3-70b-versatile on 2026-08-16 for free and
# developer plans, so the default is Groq's recommended replacement. Override it
# with GROQ_MODEL in .streamlit/secrets.toml or the environment when Groq
# retires this one too (see console.groq.com/docs/deprecations).
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.2
LLM_TIMEOUT_SECONDS = 180
AGENT_MAX_ITER = 6

FIRESTORE_COLLECTION = "code_reviews"
HISTORY_LIMIT = 25
HISTORY_CACHE_SECONDS = 300
HISTORY_PLACEHOLDER = "Select a saved review"

MAX_PROMPT_CHARS = 6000
MAX_FIELD_CHARS = 200_000  # keeps a Firestore document well under its 1 MiB limit

RATE_LIMIT_RETRIES = 6
RATE_LIMIT_MAX_WAIT_SECONDS = 90

PROMPT_PLACEHOLDER = (
    "Example: Write a Flask endpoint that accepts an image upload, validates it "
    "and stores it on disk."
)


@dataclass(frozen=True)
class AgentSpec:
    """Display information for one agent in the live status panel."""

    icon: str
    name: str
    working: str


AGENT_SPECS: Tuple[AgentSpec, ...] = (
    AgentSpec("🧑‍💻", "Software Engineer", "writing the first implementation"),
    AgentSpec("🛡️", "Application Security Specialist", "auditing the code for vulnerabilities"),
    AgentSpec("🧪", "QA & Refactoring Engineer", "fixing findings, refactoring and writing tests"),
)

# ---------------------------------------------------------------------------
# Agent and task text
# CrewAI treats {name} as a template variable, so the only braces allowed in
# these strings are the {user_request} placeholder filled in at kickoff.
# ---------------------------------------------------------------------------
ENGINEER_ROLE = "Senior Software Engineer"
ENGINEER_GOAL = "Turn the user's request into complete, correct and runnable code."
ENGINEER_BACKSTORY = (
    "You have fifteen years of production experience. You write clear, idiomatic code "
    "with docstrings, type hints where the language supports them, input validation and "
    "explicit error handling. You never leave TODOs, placeholders or pseudocode."
)

SECURITY_ROLE = "Application Security Specialist"
SECURITY_GOAL = "Find every real vulnerability in the code and explain exactly how to fix it."
SECURITY_BACKSTORY = (
    "You are a certified application security engineer who reviews code against the OWASP "
    "Top 10 and common CWE categories. You report only issues you can point to in the code, "
    "rate each one by severity and give a concrete fix. You never invent findings."
)

QA_ROLE = "QA and Refactoring Engineer"
QA_GOAL = "Ship a hardened, refactored version of the code together with tests that prove it works."
QA_BACKSTORY = (
    "You turn audited code into production-ready code. You apply every security fix, "
    "simplify the structure without changing behaviour, and write thorough automated tests "
    "that include a regression test for every vulnerability that was found."
)

ENGINEER_TASK = (
    "A user sent this coding request:\n\n"
    "{user_request}\n\n"
    "Write a complete, runnable solution.\n"
    "Rules:\n"
    "1. Use the language the user asked for. If none is named, use Python 3.\n"
    "2. Include every import, type hints where the language supports them, docstrings, "
    "input validation and explicit error handling.\n"
    "3. No placeholders, TODO comments, pseudocode or omitted sections.\n"
    "4. Prefer the standard library. Name any third-party package in a comment at the top "
    "of the file.\n"
    "Return exactly one fenced code block with a language tag, followed by a section titled "
    "'How it works' with at most five bullet points."
)
ENGINEER_OUTPUT = (
    "One fenced code block with the complete solution, then a 'How it works' section with "
    "at most five bullet points."
)

SECURITY_TASK = (
    "Audit the code written by the Software Engineer for this request:\n\n"
    "{user_request}\n\n"
    "Review for: injection (SQL, OS command, template, code), unsafe eval, exec or "
    "deserialization, path traversal, SSRF, XXE, hard-coded secrets, weak cryptography or "
    "randomness, missing authentication or authorization, missing input validation, race "
    "conditions, resource exhaustion and denial of service, sensitive data in logs or error "
    "messages, and insecure defaults or dependencies.\n"
    "Report only issues that are present in the code. If the code is clean, say so and list "
    "the checks you performed. Never invent findings.\n"
    "Write the report in Markdown with these sections, in this order:\n"
    "## Executive summary (overall risk: Critical, High, Medium, Low or None, plus two sentences)\n"
    "## Findings table (columns: ID, Severity, CWE, Location, Issue)\n"
    "## Detailed findings (for each finding: what is wrong, how it can be exploited, the "
    "vulnerable snippet, and the recommended fix with corrected code)\n"
    "## Verdict (is the code safe to ship as it is: yes or no, and why)"
)
SECURITY_OUTPUT = (
    "A Markdown security report with the sections Executive summary, Findings table, "
    "Detailed findings and Verdict."
)

QA_TASK = (
    "You receive the original code and the security audit for this request:\n\n"
    "{user_request}\n\n"
    "Produce the final version.\n"
    "1. Fix every finding in the audit, starting with the most severe.\n"
    "2. Refactor for readability, modularity and performance without changing the required "
    "behaviour. Keep type hints and docstrings.\n"
    "3. Write an automated test suite: pytest for Python, otherwise the idiomatic framework "
    "for the language. Cover normal cases, edge cases, error handling and add one regression "
    "test per audit finding.\n"
    "Return Markdown with exactly these sections, in this order:\n"
    "## Final refactored code (one fenced code block, complete and runnable)\n"
    "## Test suite (one fenced code block, complete and runnable)\n"
    "## Change log (one bullet per audit finding with its ID and the fix applied, or "
    "'No findings to fix')\n"
    "## How to run the tests (the exact shell commands)"
)
QA_OUTPUT = (
    "Markdown with the sections Final refactored code, Test suite, Change log and How to run "
    "the tests. Both code sections are complete fenced code blocks."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def get_secret(name: str, default: Any = None) -> Any:
    """Read a Streamlit secret without raising when secrets.toml is missing."""
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001  (KeyError, FileNotFoundError, secrets errors)
        return default


def clean_key(value: Any) -> str:
    """Trim whitespace and stray quotes from a pasted API key."""
    return str(value or "").strip().strip("\"'").strip()


def resolve_model_id() -> str:
    """Return the LiteLLM model string for Groq, e.g. groq/openai/gpt-oss-120b."""
    configured = get_secret("GROQ_MODEL") or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    model = str(configured).strip() or DEFAULT_GROQ_MODEL
    return model if model.startswith("groq/") else f"groq/{model}"


_GROQ_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]{8,}")


def redact(text: str, *secrets: str) -> str:
    """Remove API keys from text before it is shown on screen."""
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "[redacted]")
    return _GROQ_KEY_PATTERN.sub("gsk_[redacted]", text)


_OUTER_MARKDOWN_FENCE = re.compile(
    r"^```(?:markdown|md)[ \t]*\n(.*)\n```$", re.DOTALL | re.IGNORECASE
)


def clean_llm_text(text: Any) -> str:
    """Strip whitespace and unwrap an answer the model wrapped in a markdown fence."""
    value = str(text or "").strip()
    match = _OUTER_MARKDOWN_FENCE.match(value)
    return match.group(1).strip() if match else value


def extract_text(output: Any) -> str:
    """Get the raw text from a CrewAI TaskOutput, CrewOutput or plain string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    raw = getattr(output, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    return str(output)


def limit_text(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    """Cap a string so one Firestore field cannot exceed the document size limit."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Truncated to fit the Firestore document size limit.]"


def format_timestamp(value: Any) -> str:
    """Format a Firestore timestamp as UTC text."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "unknown time"


def format_duration(seconds: float) -> str:
    """Format seconds as '1 h 5 min', '3 min 20 s' or '42 s'."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


# ---------------------------------------------------------------------------
# Error classification (used for retries and for the messages shown to the user)
# ---------------------------------------------------------------------------
def chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and the exceptions it was raised from, without loops."""
    seen: set = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def error_text(exc: BaseException) -> str:
    """Join the type and message of every exception in the chain."""
    return " | ".join(f"{type(item).__name__}: {item}" for item in chain(exc))


def status_code(exc: BaseException) -> Optional[int]:
    """Return the HTTP status carried by an exception, if it has one."""
    for attribute in ("status_code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def is_rate_limited(exc: BaseException) -> bool:
    """True for HTTP 429 style errors, false for 'request too large' errors."""
    text = error_text(exc).lower()
    if "request too large" in text:
        return False
    codes = {status_code(item) for item in chain(exc)}
    names = " ".join(type(item).__name__.lower() for item in chain(exc))
    markers = ("rate limit", "rate_limit", "too many requests")
    return 429 in codes or "ratelimit" in names or any(marker in text.lower() for marker in markers)


_WAIT_PATTERN = re.compile(r"try again in\s+([0-9][0-9a-z.]*)", re.IGNORECASE)
_WAIT_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_wait_seconds(text: str) -> Optional[float]:
    """Parse Groq's 'Please try again in 1m3.5s' hint into seconds."""
    match = _WAIT_PATTERN.search(text)
    if not match:
        return None
    parts = _WAIT_PART.findall(match.group(1).rstrip(".,;:").lower())
    if not parts:
        return None
    return sum(float(amount) * _UNIT_SECONDS[unit] for amount, unit in parts)


def retry_after_header(exc: BaseException) -> Optional[float]:
    """Read a Retry-After header (in seconds) from the exception's HTTP response."""
    for item in chain(exc):
        headers = getattr(getattr(item, "response", None), "headers", None)
        if headers is None:
            continue
        try:
            value = headers.get("retry-after")
            if value is not None:
                return float(value)
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def seconds_to_wait(exc: BaseException, attempt: int) -> Optional[float]:
    """Seconds to sleep before retrying, or None when retrying is pointless."""
    if not is_rate_limited(exc):
        return None
    text = error_text(exc)
    lowered = text.lower()
    if "per day" in lowered or "(tpd)" in lowered or "(rpd)" in lowered:
        return None  # a daily allowance does not reset within a minute
    wait = parse_wait_seconds(text)
    if wait is None:
        wait = retry_after_header(exc)
    if wait is None:
        wait = min(10.0 * attempt, 60.0)
    if wait > RATE_LIMIT_MAX_WAIT_SECONDS:
        return None
    return float(math.ceil(wait) + 1)


def friendly_error(exc: BaseException, model_id: str) -> Tuple[str, str]:
    """Return (title, how to fix it) for the error shown to the user."""
    text = error_text(exc)
    low = text.lower()
    codes = {code for code in (status_code(item) for item in chain(exc)) if code is not None}
    model_name = model_id.removeprefix("groq/")

    if "request too large" in low or 413 in codes:
        return (
            "The request is too large for your Groq plan",
            "Groq caps the tokens one request can use per minute (8,000 on the free plan). "
            "Shorten the prompt or upgrade at console.groq.com/settings/billing.",
        )
    if is_rate_limited(exc):
        if "per day" in low or "(tpd)" in low or "(rpd)" in low:
            wait = parse_wait_seconds(text)
            when = f" Try again in about {format_duration(wait)}." if wait else ""
            return (
                "Daily Groq limit reached",
                "Your daily token or request allowance is used up." + when
                + " Upgrade at console.groq.com/settings/billing for higher limits.",
            )
        return (
            "Groq rate limit reached",
            "The crew waited and retried, but the per-minute limit was still exceeded. "
            "Wait a minute, then run again.",
        )
    if (
        any(m in low for m in ("model_not_found", "decommissioned", "model_deprecated"))
        or "does not exist or you do not have access" in low
        or 404 in codes
    ):
        return (
            f"Groq cannot serve the model {model_name}",
            "The model was retired or your key has no access to it. Pick a current model ID "
            "from console.groq.com/docs/models and set GROQ_MODEL in .streamlit/secrets.toml, "
            "for example GROQ_MODEL = \"openai/gpt-oss-120b\".",
        )
    if "invalid api key" in low or "invalid_api_key" in low or 401 in codes:
        return (
            "Groq rejected the API key",
            "Paste a valid key in the sidebar. Create one at console.groq.com/keys; "
            "it starts with gsk_.",
        )
    if 403 in codes or "forbidden" in low or "permission" in low:
        return (
            "Groq refused the request (403)",
            "The key may lack access to this model, or the request came from a blocked "
            "network such as a VPN or a shared cloud IP. Check the model permissions in your "
            "Groq console and try another network.",
        )
    if any(m in low for m in ("timed out", "timeout", "connection error", "connecterror", "network")):
        return (
            "Could not reach Groq",
            "Check your internet connection and run again. If the problem repeats, "
            "check status.groq.com.",
        )
    return (
        "The crew stopped before finishing",
        "Open the technical details below. If the message names the model or the key, fix "
        "that first. Otherwise run again.",
    )


# ---------------------------------------------------------------------------
# Firestore
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _connect_firestore(config_json: str) -> Any:
    """Create (once per credentials) a Firestore client. Raises on bad credentials."""
    config = json.loads(config_json)
    try:
        existing = firebase_admin.get_app()
    except ValueError:
        existing = None
    if existing is not None and getattr(existing, "project_id", None) != config.get("project_id"):
        firebase_admin.delete_app(existing)
        existing = None
    app = existing or firebase_admin.initialize_app(credentials.Certificate(config))
    return firestore.client(app)


def init_firestore() -> Tuple[Optional[Any], str]:
    """Return (client, message). The client is None when Firebase is not configured."""
    if firebase_admin is None:
        return None, f"firebase-admin could not be imported ({FIREBASE_IMPORT_ERROR})."
    try:
        raw_config = st.secrets["firebase"]
    except Exception:  # noqa: BLE001
        return None, "No [firebase] section found in .streamlit/secrets.toml."
    try:
        config = {str(key): value for key, value in dict(raw_config).items()}
        private_key = config.get("private_key")
        if isinstance(private_key, str):
            # Secrets pasted with literal \n sequences still produce a valid PEM key.
            config["private_key"] = private_key.replace("\\n", "\n")
        return _connect_firestore(json.dumps(config, sort_keys=True)), "Connected to Firestore."
    except Exception as exc:  # noqa: BLE001
        return None, f"Firebase setup failed: {type(exc).__name__}: {exc}"


@st.cache_data(ttl=HISTORY_CACHE_SECONDS, show_spinner=False)
def load_history(_db: Any, limit: int = HISTORY_LIMIT) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (newest reviews first, error message). Errors are cached too, so an
    unreachable database does not slow down every rerun."""
    try:
        query = (
            _db.collection(FIRESTORE_COLLECTION)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        items: List[Dict[str, Any]] = []
        for snapshot in query.stream(timeout=20):
            data = snapshot.to_dict() or {}
            items.append(
                {
                    "id": snapshot.id,
                    "prompt": str(data.get("prompt", "")),
                    "initial_code": str(data.get("initial_code", "")),
                    "security_audit": str(data.get("security_audit", "")),
                    "final_code": str(data.get("final_code", "")),
                    "timestamp": format_timestamp(data.get("timestamp")),
                }
            )
        return items, None
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def save_review(firestore_db: Any, record: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """Save one finished run to the code_reviews collection."""
    try:
        firestore_db.collection(FIRESTORE_COLLECTION).add(
            {
                "prompt": limit_text(record["prompt"]),
                "initial_code": limit_text(record["initial_code"]),
                "security_audit": limit_text(record["security_audit"]),
                "final_code": limit_text(record["final_code"]),
                "timestamp": firestore.SERVER_TIMESTAMP,
            },
            timeout=30,
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# Module-level Firestore handle: stays None when secrets are missing or invalid,
# and the rest of the app keeps working without history.
db, FIREBASE_STATUS = init_firestore()


# ---------------------------------------------------------------------------
# Live status panel
# ---------------------------------------------------------------------------
def _ui_safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """A drawing problem must never abort a crew run, so swallow ordinary errors.
    Streamlit's rerun/stop signals derive from BaseException and still pass through."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except Exception:  # noqa: BLE001
            return None

    return wrapper


class LiveStatus:
    """Draws one row per agent plus a progress bar inside an st.status container.

    Every row is an st.empty() placeholder that is redrawn as the agent moves
    through waiting, working, done or failed.
    """

    def __init__(self, status: Any, specs: Tuple[AgentSpec, ...]) -> None:
        self.status = status
        self.specs = specs
        self.progress = st.progress(0.0, text="Preparing the crew")
        self.slots = [st.empty() for _ in specs]
        self.notice = st.empty()
        self.state = ["waiting"] * len(specs)
        self.started: List[float] = [0.0] * len(specs)
        self.elapsed: List[float] = [0.0] * len(specs)
        self.steps = [0] * len(specs)
        self.errors = [""] * len(specs)
        self.active: Optional[int] = None
        for index in range(len(specs)):
            self._draw(index)

    def _draw(self, index: int) -> None:
        spec = self.specs[index]
        title = f"{spec.icon} **{spec.name}**"
        slot = self.slots[index]
        state = self.state[index]
        if state == "waiting":
            slot.markdown(f"⏳ {title}: waiting for its turn")
        elif state == "working":
            steps = self.steps[index]
            detail = f" (reasoning step {steps})" if steps else ""
            slot.info(f"🔄 {title}: {spec.working}{detail}")
        elif state == "done":
            slot.success(f"✅ {title}: finished in {self.elapsed[index]:.1f}s")
        else:
            slot.error(f"❌ {title}: {self.errors[index]}")

    @_ui_safe
    def start(self, index: int) -> None:
        self.state[index] = "working"
        self.started[index] = time.perf_counter()
        self.active = index
        self._draw(index)
        done = sum(1 for value in self.state if value == "done")
        self.progress.progress(done / len(self.specs), text=f"{self.specs[index].name} is working")
        self.status.update(label=f"{self.specs[index].name} is working", state="running")

    @_ui_safe
    def step(self, index: int) -> None:
        self.steps[index] += 1
        if self.state[index] == "working":
            self._draw(index)

    @_ui_safe
    def finish(self, index: int) -> None:
        self.state[index] = "done"
        self.elapsed[index] = time.perf_counter() - self.started[index]
        self._draw(index)
        done = sum(1 for value in self.state if value == "done")
        self.progress.progress(done / len(self.specs), text=f"{done} of {len(self.specs)} agents finished")

    @_ui_safe
    def fail(self, index: int, message: str) -> None:
        self.state[index] = "error"
        self.errors[index] = message
        self._draw(index)

    @_ui_safe
    def rate_limit_notice(self, seconds_left: int, attempt: int, total: int) -> None:
        self.notice.warning(
            f"⏳ Groq rate limit reached. Resuming in {seconds_left} s (retry {attempt} of {total})."
        )

    @_ui_safe
    def note(self, text: str) -> None:
        self.notice.info(text)

    @_ui_safe
    def clear_notice(self) -> None:
        self.notice.empty()


def install_rate_limit_retry(llm: Any, ui: LiveStatus) -> bool:
    """Make llm.call wait and retry when Groq answers 429, resuming the same step.

    The free Groq plan allows about 8,000 tokens per minute, so the second or
    third agent can hit the limit. Retrying only the failed call keeps the work
    of the agents that already finished. Returns False if the call cannot be wrapped.
    """
    original_call = llm.call

    @functools.wraps(original_call)  # keeps the original signature visible to CrewAI
    def call_with_retry(*args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            try:
                return original_call(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                wait = seconds_to_wait(exc, attempt)
                if wait is None or attempt > RATE_LIMIT_RETRIES:
                    raise
                for remaining in range(int(wait), 0, -1):
                    ui.rate_limit_notice(remaining, attempt, RATE_LIMIT_RETRIES)
                    time.sleep(1)
                ui.clear_notice()

    try:
        llm.call = call_with_retry
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# The crew
# ---------------------------------------------------------------------------
def build_crew(llm: Any, ui: LiveStatus, collected: Dict[int, str]) -> Any:
    """Create the three agents, their tasks and the sequential crew."""

    def task_done(index: int) -> Callable[[Any], None]:
        def on_done(output: Any) -> None:
            collected[index] = extract_text(output)
            ui.finish(index)
            if index + 1 < len(AGENT_SPECS):
                ui.start(index + 1)

        return on_done

    def agent_step(index: int) -> Callable[..., None]:
        def on_step(*_args: Any, **_kwargs: Any) -> None:
            ui.step(index)

        return on_step

    engineer = Agent(
        role=ENGINEER_ROLE,
        goal=ENGINEER_GOAL,
        backstory=ENGINEER_BACKSTORY,
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=AGENT_MAX_ITER,
        step_callback=agent_step(0),
    )
    security = Agent(
        role=SECURITY_ROLE,
        goal=SECURITY_GOAL,
        backstory=SECURITY_BACKSTORY,
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=AGENT_MAX_ITER,
        step_callback=agent_step(1),
    )
    qa_engineer = Agent(
        role=QA_ROLE,
        goal=QA_GOAL,
        backstory=QA_BACKSTORY,
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=AGENT_MAX_ITER,
        step_callback=agent_step(2),
    )

    write_code = Task(
        description=ENGINEER_TASK,
        expected_output=ENGINEER_OUTPUT,
        agent=engineer,
        callback=task_done(0),
    )
    audit_code = Task(
        description=SECURITY_TASK,
        expected_output=SECURITY_OUTPUT,
        agent=security,
        context=[write_code],
        callback=task_done(1),
    )
    refactor_code = Task(
        description=QA_TASK,
        expected_output=QA_OUTPUT,
        agent=qa_engineer,
        context=[write_code, audit_code],
        callback=task_done(2),
    )

    return Crew(
        agents=[engineer, security, qa_engineer],
        tasks=[write_code, audit_code, refactor_code],
        process=Process.sequential,
        verbose=False,
        memory=False,
        tracing=False,
    )


def run_crew(prompt: str, api_key: str, model_id: str, ui: LiveStatus) -> Dict[str, str]:
    """Run the three agents in order and return their outputs."""
    # Map the environment on every run, as required: LiteLLM and CrewAI helpers
    # must never fall back to an OpenAI key or endpoint.
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = "dummy-key-to-prevent-openai-errors"
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

    # api_key is also passed explicitly so concurrent sessions never share a key
    # through the process-wide environment.
    llm = LLM(
        model=model_id,
        temperature=GROQ_TEMPERATURE,
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    install_rate_limit_retry(llm, ui)

    collected: Dict[int, str] = {}
    crew = build_crew(llm, ui, collected)

    ui.start(0)
    output = crew.kickoff(inputs={"user_request": prompt})

    texts = [collected.get(index, "") for index in range(len(AGENT_SPECS))]
    task_outputs = getattr(output, "tasks_output", None) or []
    for index in range(len(AGENT_SPECS)):
        if not texts[index] and index < len(task_outputs):
            texts[index] = extract_text(task_outputs[index])
    if not texts[-1]:
        texts[-1] = extract_text(output)
    texts = [clean_llm_text(text) for text in texts]

    missing = [AGENT_SPECS[index].name for index, text in enumerate(texts) if not text]
    if missing:
        raise RuntimeError(
            "These agents returned no text: " + ", ".join(missing)
            + ". Run again. If it repeats, set a different GROQ_MODEL."
        )
    return {
        "prompt": prompt,
        "initial_code": texts[0],
        "security_audit": texts[1],
        "final_code": texts[2],
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render_sidebar_settings() -> Tuple[str, Any]:
    """Draw the API key input. Returns (api_key, container reserved for history)."""
    with st.sidebar:
        st.header("Settings")
        st.subheader("Groq API key")
        typed = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
            key="groq_key_input",
            label_visibility="collapsed",
            help="Kept only in this browser session. Leave it empty to use GROQ_API_KEY "
            "from Streamlit secrets.",
        )
        api_key = clean_key(typed)
        from_secrets = False
        if not api_key:
            api_key = clean_key(get_secret("GROQ_API_KEY"))
            from_secrets = bool(api_key)

        if from_secrets:
            st.success("Using GROQ_API_KEY from Streamlit secrets.")
        elif api_key and not api_key.startswith("gsk_"):
            st.warning("Groq keys start with gsk_. Check that you pasted the whole key.")
        elif api_key:
            st.caption("Using the key you entered.")
        else:
            st.warning(
                "No key found. Paste one above or add GROQ_API_KEY to "
                ".streamlit/secrets.toml. Create a key at console.groq.com/keys."
            )
        st.divider()
        history_container = st.container()
    return api_key, history_container


def history_label(item: Dict[str, Any]) -> str:
    """One-line label for the history dropdown."""
    prompt = " ".join(str(item.get("prompt", "")).split())
    snippet = prompt if len(prompt) <= 48 else prompt[:47].rstrip() + "…"
    return f"{item.get('timestamp', 'unknown time')}: {snippet or 'empty prompt'}"


def on_history_selected(by_id: Dict[str, Dict[str, Any]]) -> None:
    """Callback: show the selected saved review and put its prompt in the editor."""
    doc_id = st.session_state.get("history_select")
    item = by_id.get(doc_id) if doc_id else None
    if item is None:
        return
    st.session_state["result"] = {
        "prompt": item["prompt"],
        "initial_code": item["initial_code"],
        "security_audit": item["security_audit"],
        "final_code": item["final_code"],
        "source": "history",
        "when": item["timestamp"],
        "save_state": "saved",
        "save_error": None,
    }
    st.session_state["prompt_input"] = item["prompt"]


def render_history(container: Any, firestore_db: Any, status_message: str) -> None:
    """Draw the review history dropdown in the sidebar."""
    with container:
        st.subheader("Review history")
        if firestore_db is None:
            st.info(
                "History is off. Add a [firebase] section to .streamlit/secrets.toml "
                "to save and reload reviews."
            )
            st.caption(status_message)
            return

        items, error = load_history(firestore_db)
        if error:
            st.warning("Could not read the history from Firestore.")
            st.caption(error)

        by_id = {item["id"]: item for item in items}
        options: List[Optional[str]] = [None] + list(by_id.keys())
        if st.session_state.get("history_select") not in options:
            st.session_state["history_select"] = None

        def label_for(doc_id: Optional[str]) -> str:
            if doc_id is None:
                return HISTORY_PLACEHOLDER
            found = by_id.get(doc_id)
            return history_label(found) if found else str(doc_id)

        st.selectbox(
            "Load a past review",
            options=options,
            format_func=label_for,
            key="history_select",
            on_change=on_history_selected,
            args=(by_id,),
        )
        if not items and not error:
            st.caption("No saved reviews yet. Every finished run is saved here automatically.")
        st.button("Refresh list", key="refresh_history", on_click=load_history.clear)


def describe_source(result: Dict[str, Any]) -> str:
    """Caption under the results heading."""
    if result.get("source") == "history":
        return f"Loaded from history. Saved {result.get('when', 'at an unknown time')}."
    state = result.get("save_state")
    if state == "saved":
        return f"Saved to Firestore collection {FIRESTORE_COLLECTION}."
    if state == "failed":
        return f"Not saved to history: {result.get('save_error', 'unknown error')}"
    return "Not saved to history because Firestore is not configured."


def render_results(container: Any) -> None:
    """Draw the three result tabs from the session state."""
    result = st.session_state.get("result")
    with container:
        if not result:
            st.info(
                "Results appear here after a run: the initial code, the security audit "
                "and the final refactored code with tests."
            )
            return
        st.divider()
        st.subheader("Results")
        st.caption(describe_source(result))
        with st.expander("Prompt used", expanded=False):
            st.text(result["prompt"])
        tab_code, tab_audit, tab_final = st.tabs(
            ["🧑‍💻 Initial Code", "🛡️ Security Audit", "✅ Final Refactored Code & Tests"]
        )
        with tab_code:
            st.markdown(result["initial_code"])
        with tab_audit:
            st.markdown(result["security_audit"])
        with tab_final:
            st.markdown(result["final_code"])


def execute_run(prompt: str, api_key: str, model_id: str, firestore_db: Any, container: Any) -> None:
    """Validate the inputs, run the crew with live status, save and store the result."""
    prompt = (prompt or "").strip()
    with container:
        if CREWAI_IMPORT_ERROR:
            st.error("CrewAI could not be imported, so the crew cannot run. Reinstall with "
                     "pip install -r requirements.txt.")
            st.code(CREWAI_IMPORT_ERROR, language="text")
            return
        if not api_key:
            st.error("Add a Groq API key in the sidebar or set GROQ_API_KEY in "
                     ".streamlit/secrets.toml, then run again.")
            return
        if not prompt:
            st.warning("Describe what you want built, then run again.")
            return

        status = st.status("Starting the crew", expanded=True)
        result: Optional[Dict[str, str]] = None
        failure: Optional[BaseException] = None
        trace = ""
        started = time.perf_counter()
        with status:
            ui: Optional[LiveStatus] = None
            try:
                ui = LiveStatus(status, AGENT_SPECS)
                result = run_crew(prompt, api_key, model_id, ui)
            except Exception as exc:  # noqa: BLE001
                failure = exc
                trace = traceback.format_exc()
                if ui is not None:
                    ui.fail(ui.active if ui.active is not None else 0, "stopped before finishing")
            else:
                save_state, save_error = "disabled", None
                if firestore_db is not None:
                    ui.note("Saving the review to Firestore")
                    saved, save_error = save_review(firestore_db, result)
                    save_state = "saved" if saved else "failed"
                    ui.clear_notice()
                result.update({"save_state": save_state, "save_error": save_error})

        if failure is not None or result is None:
            status.update(label="The crew stopped", state="error", expanded=True)
            title, hint = friendly_error(failure or RuntimeError("No result"), model_id)
            st.error(f"**{title}.** {hint}")
            with st.expander("Technical details"):
                st.code(redact(trace or str(failure), api_key), language="text")
            return

        elapsed = format_duration(time.perf_counter() - started)
        status.update(label=f"All three agents finished in {elapsed}", state="complete", expanded=False)
        st.session_state["result"] = {
            "prompt": result["prompt"],
            "initial_code": result["initial_code"],
            "security_audit": result["security_audit"],
            "final_code": result["final_code"],
            "source": "live",
            "when": format_timestamp(datetime.now(timezone.utc)),
            "save_state": result["save_state"],
            "save_error": result["save_error"],
        }
        # The history sidebar is drawn after this function, so it picks up the
        # new review and starts from the placeholder option.
        load_history.clear()
        st.session_state["history_select"] = None


def main() -> None:
    """Lay out the page and handle the Run button."""
    api_key, history_container = render_sidebar_settings()
    model_id = resolve_model_id()

    st.title("🛡️ SentinelCoder")
    st.caption(
        "Three AI agents write your code, audit it for vulnerabilities, then refactor it "
        "and write tests."
    )
    if CREWAI_IMPORT_ERROR:
        st.error("CrewAI could not be imported. Reinstall with pip install -r requirements.txt.")

    prompt = st.text_area(
        "Coding prompt",
        key="prompt_input",
        height=180,
        max_chars=MAX_PROMPT_CHARS,
        placeholder=PROMPT_PLACEHOLDER,
    )
    run_clicked = st.button("Run SentinelCoder", type="primary", key="run_button")
    st.caption(
        f"Model: {model_id.removeprefix('groq/')} on Groq at temperature {GROQ_TEMPERATURE}."
    )

    status_container = st.container()
    results_container = st.container()

    if run_clicked:
        execute_run(prompt, api_key, model_id, db, status_container)

    render_history(history_container, db, FIREBASE_STATUS)
    render_results(results_container)


if __name__ == "__main__":
    main()
