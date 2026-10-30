# SAREMem

**Seed-Anchored Reranking and Expansion for Multimodal Agent Memory**

SAREMem couples relevance-based reranking with context expansion around direct evidence. It stores text/image memories through **Add** and retrieves original evidence through **Search**. Agent Memory Leaderboard (AML) handles Answer and Eval.

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
       → deduplicate and return up to 20 original memories
```

- **Reranking:** query-relative relevance labels create stable direct, context and other groups, preserving seed order within each group. This replaces the earlier rule that promoted a direct candidate across at most three adjacent unrelated candidates. The separate legacy numeric reranker stays disabled.
- **Expansion:** protect selected direct/context evidence, then use retained direct memories as anchors for the preceding and following memory in the same user/session. Deduplication, remaining output slots and media limits constrain expansion; it makes no extra LLM or embedding calls.

The core contribution is their coupling: the same labels control evidence priority, protection and context anchors. Neighbors may lie outside the initial seed set, but returned IDs/content always refer to original stored memories.

Candidate-local errors are skipped while the remaining assessment pool continues. Recovery follows `simple → legacy → seed`, reusing the same completed seed without new retrieval or model calls; an unavailable seed produces an error rather than a fabricated empty success.

## API and configuration

- `GET /health` needs no authentication. `POST /add` and `POST /search` accept `X-Api-Key`, `Authorization: Token ...` or `Authorization: Bearer ...`.
- Content uses ordered arrays of `text` and `image_url` parts; images are inline Base64 JPEG/PNG/WebP Data URIs. See [request/response schemas](production/aml_api/schemas.py).
- Add acknowledges only after persistence and supports idempotent writes. Search returns `{"data":[...]}`, original memory ID/content, and at most `min(top_k, 20, configured limit)` records, scoped by `user_id`.
- Defaults: legacy seed retrieval, `stable_groups`/40 assessments and session-neighbor expansion. See the [test](testing/config.env.example) and [production](production/config.env.example) templates for settings; `AML_SIMPLE_NEIGHBOR_MODE=off` disables expansion.
- Add/Search images default to request-only resizing: edge ≤2048, pixels ≤2,097,152, preserved aspect ratio, no crop/enlargement. Stored originals stay intact. `AML_ADD_IMAGE_MODE=off` and `AML_SEARCH_IMAGE_MODE=off` disable their respective components.
- Each Add/Search request has a maximum 1680-second application budget; provider read timeout is 300 seconds and connect/write/pool limits are 30 seconds, clipped to remaining time. Add computation is isolated; the parent atomically commits results. Capacity exhaustion returns 429 with Retry-After.

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

The template listens on `127.0.0.1:8000`; check `/health` from another terminal. Use a supervisor for persistent hosting and configure public routing/TLS and gateway timeouts for deployment. Production emits no per-request diagnostics; startup makes no model calls. Keep real keys, databases and test outputs out of Git.

For official evaluation, register endpoints, capacity and a fixed commit under the [AML instructions](https://agentmemoryleaderboard.ai/api-guide). Local smoke is separate from organizer-run evaluation; never index gold answers or clues. Connectivity tests do not establish full-scale capacity or general retrieval gains.

Verify distributed files with `sha256sum -c SHA256SUMS` (`shasum -a 256 -c SHA256SUMS` on macOS).
