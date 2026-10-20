# aitrainingplan-agent

Athlete profile in, half-marathon plan out as an `.ics` calendar, with a code-enforced safety check between the two LLM agents.

## Workflow

1. **Planner (LLM)**: `athlete.json` → 12-week plan as JSON.
2. **Validator (plain code)**: checks the coaching rules (start ≤110% of current volume, ≤+10% per week, a deload every 4th week, a taper, long run share, 80/20 easy/hard, no back-to-back hard days, days available). It returns a list of violations. It is code, not an LLM, because these rules are arithmetic.
3. **Repairer (LLM)**: gets athlete + plan + violations and returns the full corrected plan. Goes back to step 2, max 3 rounds, then aborts.
4. **Publish**: `plan.json` + `plan.ics`. Every step's input, output, timing and token usage goes to `runs/<timestamp>/log.jsonl`.

## Run

```
# .env: ANTHROPIC_API_KEY=...  or  OPENAI_API_KEY=... + LLM_BASE_URL=https://api.openai.com/v1 + MODEL=gpt-4.1
uv run --env-file .env pipeline.py [athlete.json]
```

## Where it broke

First live run (gpt-4.1), `runs/20261020-210935/`: aborted after 3 repairs, with violations going 15 → 13 → 8 → 4. Each repair shortened the long runs to meet a flat "long run ≤35% of the week" rule, and that broke the hard-km share or the +10% ramp somewhere else. The cause was the validator, not the model: a 4-day runner doing 25 km with a 10 km long run is already at 40%, so the rules could not all be satisfied. The fix (`465122e`) scales the cap with training days: 45% for ≤4 days, 40% for 5, 35% for 6+. Next run, `runs/20261020-211302/`: 0 violations on the first attempt, published.

Still open: the repair loop only reports *that* it didn't converge. It should flag rules that keep firing across rounds as likely unsatisfiable, and keep the best attempt.
