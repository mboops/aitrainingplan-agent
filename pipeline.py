# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27"]
# ///
"""plan -> validate -> repair (loop) -> publish, for aitrainingplan.app.

The planner and the repairer are LLM agents. The validator is plain code on purpose:
training-load rules are arithmetic, and arithmetic is what LLMs get wrong while
sounding confident. Every step's input and output is written to runs/<timestamp>/.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HARD_TYPES = {"tempo", "intervals", "race"}
EASY_TYPES = {"easy", "long", "recovery"}
RUN_TYPES = HARD_TYPES | EASY_TYPES
TYPES = RUN_TYPES | {"strength", "rest"}
MAX_REPAIRS = 3

PLAN_SCHEMA = """{"weeks": [{"week": 1, "sessions": [
  {"day": "Mon|Tue|Wed|Thu|Fri|Sat|Sun",
   "type": "easy|long|recovery|tempo|intervals|race|strength|rest",
   "distance_km": 8.0,
   "intensity": "easy|hard",
   "description": "what to do, with paces as effort (RPE) not min/km"}]}]}"""

PLANNER_SYSTEM = f"""You are an endurance running coach writing a plan for a self-coached athlete.
Return ONLY a JSON object matching this schema, no prose:
{PLAN_SCHEMA}
Principles: progressive overload, a lighter week every 4th week, a taper before race day,
most running easy, never two hard days in a row, respect the days the athlete has."""

REPAIR_SYSTEM = f"""You fix training plans that failed automated safety checks.
You receive the athlete, the plan, and a list of violations. Return the FULL corrected plan
as JSON matching this schema, no prose:
{PLAN_SCHEMA}
Fix every violation. Change as little as possible: do not rewrite sessions that are fine.
Recompute weekly totals after each change - fixing one week often breaks the next."""


# ---------- LLM ----------


def call_llm(system: str, user: str) -> tuple[str, dict]:
    """Anthropic if ANTHROPIC_API_KEY is set, else any OpenAI-compatible endpoint (DeepSeek default)."""
    t0 = time.monotonic()
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("MODEL", "claude-sonnet-5-5")
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 16000,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=300,
        )
        r.raise_for_status()
        body = r.json()
        text = "".join(b.get("text", "") for b in body["content"])
        usage = body.get("usage", {})
    else:
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ["OPENAI_API_KEY"]
        base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        model = os.environ.get("MODEL", "deepseek-chat")
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "max_tokens": 8000,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=300,
        )
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
    return text, {
        "model": model,
        "seconds": round(time.monotonic() - t0, 1),
        "usage": usage,
    }


def parse_plan(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return json.loads(m.group(1) if m else text)


# ---------- validator (deterministic) ----------


def week_km(week: dict) -> float:
    return sum(s["distance_km"] for s in week["sessions"] if s["type"] in RUN_TYPES)


def validate(plan: dict, athlete: dict) -> list[str]:
    v: list[str] = []
    weeks = plan.get("weeks")
    n = athlete["weeks"]
    if not isinstance(weeks, list) or len(weeks) != n:
        return [
            f"plan must have exactly {n} weeks, got {len(weeks) if isinstance(weeks, list) else 'none'}"
        ]

    # schema and per-session sanity
    for w in weeks:
        for s in w.get("sessions", []):
            where = f"week {w.get('week')} {s.get('day')}"
            if s.get("day") not in DAYS:
                v.append(f"{where}: day must be one of {DAYS}")
            if s.get("type") not in TYPES:
                v.append(f"{where}: unknown type {s.get('type')!r}")
            if (
                not isinstance(s.get("distance_km"), (int, float))
                or s["distance_km"] < 0
            ):
                v.append(f"{where}: distance_km must be a non-negative number")
            if s.get("type") in HARD_TYPES and s.get("intensity") != "hard":
                v.append(f"{where}: {s['type']} must be intensity 'hard'")
            if s.get("type") in EASY_TYPES and s.get("intensity") != "easy":
                v.append(f"{where}: {s['type']} must be intensity 'easy'")
    if v:
        return v  # numbers below are meaningless on a malformed plan

    vols = [week_km(w) for w in weeks]
    deloads = {
        i for i in range(3, n - 2, 4)
    }  # 0-based: weeks 4, 8, ... outside the taper
    taper = {n - 2, n - 1}

    for i, w in enumerate(weeks):
        wk = i + 1
        active = [s for s in w["sessions"] if s["type"] != "rest"]
        if len(active) > athlete["days_per_week"]:
            v.append(
                f"week {wk}: {len(active)} training days, athlete has {athlete['days_per_week']}"
            )
        days = [s["day"] for s in active]
        if len(days) != len(set(days)):
            v.append(f"week {wk}: two sessions on the same day")

        if i == 0:
            cap = athlete["current_weekly_km"] * 1.10
            if vols[0] > cap:
                v.append(
                    f"week 1: {vols[0]:.1f} km exceeds 110% of current {athlete['current_weekly_km']} km ({cap:.1f})"
                )
        elif i in deloads:
            if vols[i] > vols[i - 1] * 0.80:
                v.append(
                    f"week {wk}: deload week must be <=80% of week {wk - 1} ({vols[i]:.1f} vs cap {vols[i - 1] * 0.8:.1f} km)"
                )
        elif i not in taper:
            ref = (
                i - 2 if (i - 1) in deloads else i - 1
            )  # after a deload, compare with the week before it
            cap = vols[ref] * 1.10
            if vols[i] > cap + 0.05:
                v.append(
                    f"week {wk}: {vols[i]:.1f} km is more than +10% on week {ref + 1} ({vols[ref]:.1f} km, cap {cap:.1f})"
                )

        if i < n - 1 and vols[i] > 0:
            longest = max(
                (s["distance_km"] for s in w["sessions"] if s["type"] in RUN_TYPES),
                default=0,
            )
            if longest > 0.35 * vols[i] + 0.05:
                v.append(
                    f"week {wk}: longest run {longest:.1f} km is over 35% of the week ({vols[i]:.1f} km)"
                )
            hard = sum(
                s["distance_km"] for s in w["sessions"] if s["type"] in HARD_TYPES
            )
            if hard > 0.25 * vols[i] + 0.05:
                v.append(
                    f"week {wk}: {hard:.1f} km hard is over 25% of {vols[i]:.1f} km (80/20 with warm-up slack)"
                )

    peak = max(vols[: n - 2])
    if vols[n - 2] > 0.80 * peak:
        v.append(
            f"week {n - 1}: taper must be <=80% of peak ({vols[n - 2]:.1f} vs cap {0.8 * peak:.1f} km)"
        )
    race_week_other = vols[n - 1] - athlete["race_km"]
    if race_week_other > 0.40 * peak:
        v.append(
            f"week {n}: {race_week_other:.1f} km besides the race is over 40% of peak ({0.4 * peak:.1f})"
        )

    races = [s for s in weeks[-1]["sessions"] if s["type"] == "race"]
    race_day = DAYS[date.fromisoformat(athlete["race_date"]).weekday()]
    if (
        len(races) != 1
        or races[0]["day"] != race_day
        or abs(races[0]["distance_km"] - athlete["race_km"]) > 0.2
    ):
        v.append(
            f"week {n}: needs exactly one race session of {athlete['race_km']} km on {race_day}"
        )

    # hard days in a row, across week boundaries too (Sun -> Mon)
    hard_days = sorted(
        (w["week"] - 1) * 7 + DAYS.index(s["day"])
        for w in weeks
        for s in w["sessions"]
        if s["type"] in HARD_TYPES
    )
    for a, b in pairwise(hard_days):
        if b - a == 1:
            v.append(
                f"hard sessions on consecutive days: week {a // 7 + 1} {DAYS[a % 7]} and week {b // 7 + 1} {DAYS[b % 7]}"
            )
    return v


# ---------- publish ----------


def to_ics(plan: dict, athlete: dict) -> str:
    race = date.fromisoformat(athlete["race_date"])
    start = (
        race - timedelta(days=race.weekday()) - timedelta(weeks=athlete["weeks"] - 1)
    )  # Monday of week 1
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//aitrainingplan.app//agent pipeline//EN",
    ]
    for w in plan["weeks"]:
        for s in w["sessions"]:
            if s["type"] == "rest":
                continue
            d = start + timedelta(weeks=w["week"] - 1, days=DAYS.index(s["day"]))
            title = (
                f"{s['type'].title()} {s['distance_km']:g} km"
                if s["distance_km"]
                else s["type"].title()
            )
            desc = (
                s.get("description", "")
                .replace("\\", "\\\\")
                .replace(",", "\\,")
                .replace(";", "\\;")
                .replace("\n", "\\n")
            )
            lines += [
                "BEGIN:VEVENT",
                f"UID:w{w['week']}-{s['day']}-{s['type']}@aitrainingplan.app",
                f"DTSTAMP:{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
                f"DTSTART;VALUE=DATE:{d:%Y%m%d}",
                f"SUMMARY:{title}",
                f"DESCRIPTION:{desc}",
                "END:VEVENT",
            ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# ---------- orchestration ----------


def main(athlete_path: str = "athlete.json") -> int:
    athlete = json.loads(Path(athlete_path).read_text())
    run = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    log = run / "log.jsonl"

    def record(step: str, **fields):
        entry = {"step": step, **fields}
        with log.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"[{step}] "
            + ", ".join(f"{k}={v}" for k, v in fields.items() if k != "violations")
        )

    text, meta = call_llm(PLANNER_SYSTEM, "Athlete:\n" + json.dumps(athlete, indent=2))
    (run / "01_planner_raw.txt").write_text(text)
    record("plan", **meta)

    plan = None
    for attempt in range(MAX_REPAIRS + 1):
        try:
            plan = parse_plan(text)
            violations = validate(plan, athlete)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            violations = [f"output is not a valid plan: {type(e).__name__}: {e}"]
        (run / f"02_validation_{attempt}.json").write_text(
            json.dumps(violations, indent=2)
        )
        record(
            "validate",
            attempt=attempt,
            violations_count=len(violations),
            violations=violations,
        )
        for line in violations:
            print("   -", line)
        if not violations:
            break
        if attempt == MAX_REPAIRS:
            record(
                "abort",
                reason=f"still {len(violations)} violations after {MAX_REPAIRS} repairs",
            )
            return 1
        user = json.dumps(
            {"athlete": athlete, "plan": plan, "violations": violations}, indent=2
        )
        text, meta = call_llm(REPAIR_SYSTEM, user)
        (run / f"03_repair_{attempt + 1}_raw.txt").write_text(text)
        record("repair", attempt=attempt + 1, **meta)

    (run / "plan.json").write_text(json.dumps(plan, indent=2))
    (run / "plan.ics").write_text(to_ics(plan, athlete), newline="")
    vols = [round(week_km(w), 1) for w in plan["weeks"]]
    record("publish", weekly_km=vols, ics=str(run / "plan.ics"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
