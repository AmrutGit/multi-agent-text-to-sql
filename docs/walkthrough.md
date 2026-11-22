# Code walkthrough

One section per module that changed. Each covers what it does, why it is built this
way, the alternative that was rejected, one failure case and how it is handled, and
likely interview questions with short answers.

---

## `src/llm.py`: provider layer

**What it does.** Exposes one function, `complete(system, user) -> str`. A
`ModelConfig` picks the provider (`gemini`, `anthropic`, `openai_compat`), the model ID,
sampling and reasoning settings, and prices. Every call appends
`{in_tokens, out_tokens, seconds}` to the active `track()` scope, which the benchmark
turns into latency and cost per question.

**Why this design.** Agents depend on one function, so tests patch one name and a new
provider is one function. Gemini uses the `google-genai` SDK. Anthropic and
OpenAI-compatible servers use plain HTTPS, so a locally served fine-tuned model plugs in
the same way as an API model.

**Changes made in this pass:**
- Migrated Gemini from the deprecated `google-generativeai` package to `google-genai`
  (`genai.Client(...).models.generate_content`).
- `temperature=None` now means "do not send the field". Claude Haiku 5.5 returns HTTP
  400 for any non-default sampling value, so with the old hard-coded `temperature: 0`
  every Haiku call would have failed and been recorded as `llm_error`.
- New `effort` field, sent as `output_config.effort` (Anthropic) or `thinking_level`
  (Gemini). Both current models think by default, and the level is set explicitly so a
  provider-side default change cannot silently change results.
- Thinking tokens count as output. Gemini reports them separately
  (`thoughts_token_count`), and leaving them out would understate cost.
- An Anthropic `stop_reason: "refusal"` (HTTP 200 with no answer) now raises, so it is
  counted as `llm_error` rather than as an empty SQL string.
- API keys are read from the environment first, then from `./.env`. `GEMINI_API_KEY`
  and `GOOGLE_API_KEY` are accepted interchangeably.

**Alternative rejected.** Using each vendor's SDK for everything, or a framework such as
LangChain. Either would hide request shapes and add dependencies. The fake-HTTP-server
tests check the exact JSON body sent, which is how the `temperature` problem is pinned.

**Failure case.** Gemini returns a candidate with no text (safety block, or the output
budget used up by thinking). `resp.text` is `None`, and the code raises
`RuntimeError("LLM returned no text ...: <finish_reason>")`. The benchmark records this
as `llm_error` with the finish reason, which separates infrastructure failures from
wrong SQL.

**Interview questions**
1. *Why not temperature 0 for a benchmark?* Haiku 5.5 rejects it, and running Gemini the
   same way keeps the two models comparable. The cost is that runs are not repeatable
   bit for bit, so run-to-run variance is a listed limitation; it would be measured by
   repeating a run with the same seed.
2. *Why count thinking tokens?* Both providers bill them as output. Leaving them out
   would make $/question look several times cheaper than the invoice.
3. *Why `max_tokens` 8192 rather than 1024?* With thinking on, the limit covers thinking
   plus the answer. A low limit truncates before the SQL is written, and that would
   show up as `no_sql` failures that are really configuration errors.

---

## `evals/benchmark.py`: `run_with_budget` and `--max-usd`

**What it does.** Runs questions in batches of `--workers`. After each batch it adds up
the measured `cost_usd`, and it stops before the next batch once the total reaches
`--max-usd`. The result file's `_meta` records `spend_usd` and `stopped_by_budget`, and
`evals.report` prints a note under any run that was stopped.

**Why this design.** The cost estimate depends on how many thinking tokens a model
produces, which is unknown before the run. A hard cap based on measured spend makes the
budget safe whatever the estimate. Batching bounds the overshoot to one batch, which is
one question at `--workers 1`.

**Alternative rejected.** A pre-run estimate only. It cannot account for retries or
thinking length, and an estimate that is off by 3x on 800 calls is real money.

**Failure case.** A model without prices (`price_in: null`) reports `cost_usd = None`.
The guard treats that as $0 and never stops, which is right for a local model and is
covered by a test.

**Interview questions**
1. *Is a budget-stopped run still valid?* It is a valid smaller sample, but not the
   stratified n that was planned, and the report labels it as stopped.
2. *Why not stop mid-batch?* Requests in flight are already paid for. Waiting for the
   batch keeps every record complete.
3. *Where is cumulative spend across runs tracked?* `spend_usd` is stored in each
   result file's metadata, and the runs' totals sum to the job total.
