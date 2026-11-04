# SAREMem

**Seed-Anchored Reranking and Expansion for Multimodal Agent Memory**

SAREMem couples relevance-based reranking with context expansion around direct evidence. It stores text/image memories through **Add** and retrieves original evidence through **Search**. An optional context-sufficiency gate checks the complete final evidence set before it is returned. Agent Memory Leaderboard (AML) handles Answer and Eval.

Author: [Xinyu Sheng](https://github.com/xinyusheng-04) · [MIT License](LICENSE)

## Requirements

- Python **3.12** and a POSIX shell on Linux or macOS; use Linux for production.
- Pinned dependencies: [production](production/requirements.lock.txt) and [testing](testing/requirements.lock.txt). Testing includes runtime dependencies.
- Real requests require OpenAI-compatible providers for **gpt-4o-mini** and **text-embedding-v4**. No local GPU or model download is needed.
- Offline tests need no model credentials or dataset; local loopback listeners must be permitted.

## Quick start

Run commands from the repository root.

### 1. Install and test offline

```sh
python3.12 -m venv .venv-test
.venv-test/bin/python -m pip install -r testing/requirements.lock.txt
.venv-test/bin/python -B testing/run.py offline
```

Expected: pytest passes and exits with code `0`; no external model calls are made.

### 2. Configure an isolated API instance

```sh
mkdir -p config state/testing testing/outputs
test -e config/testing.env || cp testing/config.env.example config/testing.env
chmod 600 config/testing.env
```

Edit `config/testing.env`: replace every `CHANGE_ME`, set both providers' API keys/base URLs, and choose a random `AML_MEMORY_SYSTEM_KEY` of at least 24 characters. Use plain `AML_NAME=value` lines **without quotes or inline comments**. Relative storage paths resolve from the config file; keep the template's test database, pipeline settings and port `8001`.

On small hosts, add these limits to both test and production configurations:

```text
AML_PROD_MAX_CONCURRENT_SEARCHES=1
AML_PROD_WORKER_MEMORY_MB=1024
```

Initialize only an absent database:

```sh
test -e state/testing/memory.sqlite3 || \
  .venv-test/bin/python -B production/init_db.py --path "$(pwd)/state/testing/memory.sqlite3"
```

### 3. Start the API

Terminal A:

```sh
.venv-test/bin/python -B testing/testing_server.py \
  --config config/testing.env \
  --output-dir "testing/outputs/local-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

Terminal B, from the same repository root:

```sh
.venv-test/bin/python -B testing/run.py health \
  --config config/testing.env --allow-network
.venv-test/bin/python -B testing/run.py smoke \
  --config config/testing.env --allow-network --allow-paid --isolated-target
```

Both checks should report `status: passed`. Smoke uses real providers and can incur charges: it writes a synthetic memory, repeats Add to check idempotency, then checks Search, authentication and user isolation. Keep it pointed at the isolated instance. Optional provider-only check: `testing/run.py provider --config config/testing.env --allow-network --allow-paid`, using the same interpreter.

Stop terminal A with Ctrl-C. Reuse its database to retain memories; each test launch needs a new diagnostic output directory.

## Method

```text
Add    → captions + textual views → embeddings → atomic persistence
Search → up to 80 seed candidates → assess up to 40 candidates
       → rerank: direct / context / other → select within top-k and media limits
       → protect direct/context → expand around retained direct evidence
       → deduplicate and project up to 20 original memories
       → optional sufficiency gate → original evidence or an empty result
```

- **Reranking:** query-relative relevance labels create stable direct, context and other groups, preserving seed order within each group. This replaces the earlier rule that promoted a direct candidate across at most three adjacent unrelated candidates. The separate legacy numeric reranker stays disabled.
- **Expansion:** protect selected direct/context evidence, then use retained direct memories as anchors for the preceding and following memory in the same user/session. Deduplication, remaining output slots and media limits constrain expansion; it makes no extra LLM or embedding calls.

**Context sufficiency (optional):** after expansion and final response projection, a multimodal model assesses the entire evidence set together with the query. This is a set-level score rather than an average of per-memory relevance scores. See the configuration below.

The core contribution is their coupling: the same labels control evidence priority, protection and context anchors. Neighbors may lie outside the initial seed set, but returned IDs/content always refer to original stored memories.

Candidate-local errors are skipped while the remaining assessment pool continues. Recovery follows `simple → legacy → seed`, reusing the same completed seed without new retrieval or model calls; an unavailable seed produces an error rather than a fabricated empty success.

## API and configuration

- `GET /health` needs no authentication. `POST /add` and `POST /search` accept `X-Api-Key`, `Authorization: Token ...` or `Authorization: Bearer ...`.
- Plain text content/query may be supplied as a string. Multimodal content uses ordered arrays of `text` and `image_url` parts; images are inline Base64 JPEG/PNG/WebP Data URIs. See [request/response schemas](production/aml_api/schemas.py).
- Add acknowledges only after persistence and supports idempotent writes. Search returns `{"data":[...]}`, original memory ID/content, and at most `min(top_k, 20, configured limit)` records, scoped by `user_id`.
- Search `created_at` is the earliest valid source-message `timestamp` from the Add request, formatted as UTC ISO 8601. Input timestamps are Unix milliseconds; zero is valid. If no reliable source timestamp is supplied, `created_at` is omitted (not `null`); it never falls back to ingestion time, message order, or dates inferred from content. This is message time, not necessarily event time. All retrieval and recovery paths use the same rule; legacy stored ingestion timestamps are ignored. The existing database schema is retained, with an internal empty string for missing time on new writes.
- Defaults: legacy seed retrieval, `stable_groups`/40 assessments and session-neighbor expansion. See the [test](testing/config.env.example) and [production](production/config.env.example) templates for settings; `AML_SIMPLE_NEIGHBOR_MODE=off` disables expansion.
- Add/Search images default to request-only resizing: edge ≤2048, pixels ≤2,097,152, preserved aspect ratio, no crop/enlargement. Stored originals stay intact. `AML_ADD_IMAGE_MODE=off` and `AML_SEARCH_IMAGE_MODE=off` disable their respective components.
- Each Add/Search request has a maximum 1680-second application budget; provider read timeout is 300 seconds and connect/write/pool limits are 30 seconds, clipped to remaining time. Add computation is isolated; the parent atomically commits results. Capacity exhaustion returns 429 with Retry-After.

### Optional context sufficiency

The templates keep this component **off**. To enable it, add these lines to your private instance configuration:

```text
AML_SUFFICIENCY_MODE=on
AML_SUFFICIENCY_THRESHOLD=0.20
AML_SUFFICIENCY_MAX_SECONDS=90
```

The [component](production/aml_api/context_sufficiency/__init__.py) assesses the complete final evidence set after retrieval, reranking, expansion and response projection, including successful recovery paths. It uses **gpt-4o-mini** with the query and the original text/image evidence. The model returns integer points from 0 to 100 and a diagnostic reason; points are divided by 100 for a 0–1 score with 0.01 resolution. This is an ordinal assessment, not a calibrated probability. No citation field or citation requirement is used.

A score **below 0.20** returns `{"data":[]}`; **0.20 or higher** preserves the original evidence. The reason does not override the score. Internal scores/reasons are not added to the HTTP response. Empty evidence skips the extra call. Each nonempty Search normally adds one LLM call; bounded protocol/transport retries can add calls. Eligible failures that map to service HTTP 503 allow up to three sufficiency computations within the shared deadline; after three consecutive failures, the original revalidated evidence is returned (see Sufficiency recovery below). Other failures retain their error responses, and timeouts return 504. Errors are not treated as low-score refusals and do not start another retrieval.

The 90-second stage reservation stays inside the existing 1680-second Search budget. Under the shipped policy this reduces the effective initial simple-stage cap from 1140 to 1065 seconds; Add is unaffected. Set `AML_SUFFICIENCY_MODE=off` to restore the original Search behavior and timing. For physical removal, disable it, remove the component directory and corresponding tests, and rebuild `MANIFEST.json`/`SHA256SUMS` using the packager before distribution.

Returning empty evidence does not force the organizer's Answer to refuse. This repository implements Add/Search; local gate decisions are not official Answer or evaluation results.

## Production

Use a separate environment, key and database. Copy [production/config.env.example](production/config.env.example) to `config/production.env` only if absent, replace its placeholders, apply the host limits above, and keep it private.

```sh
python3.12 -m venv .venv-prod
.venv-prod/bin/python -m pip install -r production/requirements.lock.txt
mkdir -p state/production
chmod 600 config/production.env
test -e state/production/memory.sqlite3 || \
  .venv-prod/bin/python -B production/init_db.py --path "$(pwd)/state/production/memory.sqlite3"
.venv-prod/bin/python -B production/server.py --config config/production.env
```

The template listens on `127.0.0.1:8000`; check `/health` from another terminal. Use a supervisor for persistent hosting and configure public routing/TLS and gateway timeouts for deployment. Production HTTP responses do not expose internal diagnostics. With diagnostics enabled, the server keeps private failure and sufficiency audit logs without full request/response contents; startup makes no model calls. Keep real keys, databases and test outputs out of Git.

For official evaluation, register endpoints, capacity and a fixed commit under the [AML instructions](https://agentmemoryleaderboard.ai/api-guide). Local smoke is separate from organizer-run evaluation; never index gold answers or clues. Connectivity tests do not establish full-scale capacity or general retrieval gains.

Verify distributed files with `sha256sum -c SHA256SUMS` (`shasum -a 256 -c SHA256SUMS` on macOS).

### Execution concurrency

`AML_PROD_MAX_CONCURRENT_ADDS` and `AML_PROD_MAX_CONCURRENT_SEARCHES` independently bound active work; the templates use 16 and 8. Adjust concurrency according to server load. `AML_PROD_MAX_INFLIGHT` includes waiting requests. Add locks are per request ID, allowing different requests to compute concurrently while SQLite claim/commit transactions remain atomic. Same-ID payload conflicts and cached replay are unchanged. Concurrency needs sufficient memory and provider capacity; these limits do not guarantee throughput. With parallel Add, insertion sequence follows completion order, not source-message chronology; use reliable source timestamps where available. Keep Add and Search evaluation phases separate to avoid changing the indexed source during a Search.


### Sufficiency recovery

Within one Search request, three consecutive eligible sufficiency computation
failures that map to service HTTP 503 return the already retrieved, revalidated
evidence unchanged. This is a change to Search failure behavior. Successful
low-score decisions still return empty evidence. Timeouts, cancellation, scope,
source integrity and evidence-validation failures do not enable recovery.
Each computation has at most 90 seconds. The three computations together have
at most 270 seconds, capped by the remaining 1,680-second Search budget. The
existing 90-second retrieval reservation and retrieval stage limits are unchanged;
recovery uses available remaining time rather than reserving 270 seconds upfront.
Each computation retains the existing finite transport/protocol retry policy
(at most three sends per computation, nine overall, budget permitting).
No retrieval, answer generation, ingestion, schema migration or counter across
requests is added. Private `sufficiency.jsonl` audit records contain timestamps,
correlation, attempt counts and outcomes, without request/response contents.
