# SAREMem

**Seed-Aware Reordering of Evidence for Multimodal Memory**

SAREMem is a multimodal memory service for Agent Memory Leaderboard (AML). It stores ordered text and image memories through **Add**, and returns ranked original evidence through **Search**. AML runs the downstream Answer and Eval stages.

Author and maintainer: [Xinyu Sheng](https://github.com/xysheng-04) · [MIT License](LICENSE)

## Getting started

| Goal | Start here | Requirements |
| --- | --- | --- |
| Check the implementation offline | [Offline quick start](#offline-quick-start) | Python 3.12; no model credentials |
| Run a real Add/Search example | [Local API quick start](#local-api-quick-start) | LLM and embedding provider credentials |
| Deploy the competition service | [Production deployment](#production-deployment) | Separate production configuration and database |
| Inspect the retrieval design | [Method](#method) | Source code |

### Requirements

- Python **3.12** and a POSIX shell on Linux or macOS. Use Linux for production deployment.
- Network access to install the pinned Python dependencies. Offline tests make no external model calls after installation.
- For real requests: OpenAI-compatible endpoints serving `gpt-4o-mini` and `text-embedding-v4`. The service calls remote providers and requires no local GPU or model download.

Run all commands from the repository root, the directory containing `production/`, `testing/` and this README. With a ZIP distribution, extract it and enter `SAREMem/` first. For a Git checkout, record `git rev-parse HEAD` and use that commit throughout validation.

## Offline quick start

Install one test environment and run the included regression suite:

```sh
python3.12 -m venv .venv-test
.venv-test/bin/python -m pip install -r testing/requirements.lock.txt
.venv-test/bin/python -B testing/run.py offline
```

The test requirements include the production dependencies. This path needs no configuration file, provider key, existing database or benchmark download. Tests create synthetic data in temporary locations and use mocked providers. Local TCP binding must be permitted for the loopback HTTP checks.

**Success:** pytest reports `passed` with no failed tests, and the command exits with code `0`. The suite checks Add idempotency, user isolation, response contracts, recovery, retries, deadlines, worker cleanup and distribution startup.

## Local API quick start

This example uses an isolated test database and real providers. Add, Search and the provider probe can incur charges. Complete the offline setup above first.

### 1. Prepare a test configuration

For a fresh checkout:

```sh
mkdir -p config state/testing testing/outputs
chmod 700 config state/testing testing/outputs
test -e config/testing.env || cp testing/config.env.example config/testing.env
chmod 600 config/testing.env
.venv-test/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Use the generated random value as `AML_MEMORY_SYSTEM_KEY`. Edit `config/testing.env` and replace the four provider settings and the service key:

```text
AML_MEMORY_SYSTEM_KEY=YOUR_GENERATED_SERVICE_KEY
AML_LLM_API_KEY=YOUR_LLM_PROVIDER_KEY
AML_LLM_BASE_URL=https://YOUR_LLM_PROVIDER/v1
AML_EMBEDDING_API_KEY=YOUR_EMBEDDING_PROVIDER_KEY
AML_EMBEDDING_BASE_URL=https://YOUR_EMBEDDING_PROVIDER/v1
```

These are placeholders: enter your own values without surrounding quotes. Keep `AML_SEARCH_PIPELINE=simple`, `AML_STORAGE_PATH=../state/testing/memory.sqlite3`, port `8001` and `AML_TEST_BASE_URL=http://127.0.0.1:8001`. All `CHANGE_ME` values must be replaced. The LLM and embedding endpoints may belong to different providers.

For a small host, also add these settings to the test configuration:

```text
AML_PROD_MAX_CONCURRENT_SEARCHES=1
AML_PROD_WORKER_MEMORY_MB=1024
```

### 2. Check provider access and initialize the test database

```sh
.venv-test/bin/python -B testing/run.py provider \
  --config config/testing.env --allow-network --allow-paid
```

Success prints `{"test": "provider", "status": "passed"}`. This sends one small LLM request and one embedding request; it checks connectivity before you start the service.

Create the test database only if it is absent:

```sh
test -e state/testing/memory.sqlite3 || \
  .venv-test/bin/python -B production/init_db.py --path "$(pwd)/state/testing/memory.sqlite3"
```

On first creation, the initializer prints `Initialized a new empty AML database. No provider requests made.` It refuses to overwrite an existing file. Keep existing databases intact.

### 3. Start the API — terminal A

```sh
.venv-test/bin/python -B testing/testing_server.py \
  --config config/testing.env \
  --output-dir "testing/outputs/local-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

Leave this terminal running. The launcher is intentionally quiet; use Health to check readiness. Each launch requires a new output directory. Sanitized stage diagnostics are written to `requests.jsonl` inside that directory.

### 4. Verify the complete flow — terminal B

Open another terminal in the same repository root:

```sh
.venv-test/bin/python -B testing/run.py health \
  --config config/testing.env --allow-network
.venv-test/bin/python -B testing/run.py smoke \
  --config config/testing.env --allow-network --allow-paid --isolated-target
```

Successful output:

```text
{"test": "health", "status": "passed"}
{"test": "smoke", "status": "passed"}
```

Smoke creates a fresh synthetic user, writes a bicycle memory, repeats Add to check idempotency, retrieves evidence, and checks authentication and user isolation. It uses the same production app and retrieval service. `--isolated-target` confirms that you selected a dedicated test instance; verify the configured URL before running it.

Stop the service with Ctrl-C in terminal A and let active workers finish cleanup. Restart with the same configuration and database to retain the stored memories.

### 5. Inspect Add and Search manually (optional)

While the test service is running, enter its `AML_MEMORY_SYSTEM_KEY` through a hidden prompt in terminal B. The examples below send additional real requests.

```sh
SAREMEM_DEMO_KEY="$(.venv-test/bin/python -c 'import getpass; print(getpass.getpass("Test Memory System Key: "))')"
SAREMEM_DEMO_USER="readme-demo-$(date -u +%Y%m%dT%H%M%SZ)-$$"

curl --fail-with-body --max-time 1800 http://127.0.0.1:8001/add \
  -H 'Content-Type: application/json' \
  -H "X-Api-Key: $SAREMEM_DEMO_KEY" \
  --data-binary "{\"request_id\":\"$SAREMEM_DEMO_USER:write:1\",\"user_id\":\"$SAREMEM_DEMO_USER\",\"session_id\":\"$SAREMEM_DEMO_USER:session\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"The bicycle is blue.\"}]}]}"

curl --fail-with-body --max-time 1800 http://127.0.0.1:8001/search \
  -H 'Content-Type: application/json' \
  -H "X-Api-Key: $SAREMEM_DEMO_KEY" \
  --data-binary "{\"query\":[{\"type\":\"text\",\"text\":\"What color is the bicycle?\"}],\"user_id\":\"$SAREMEM_DEMO_USER\",\"top_k\":20}"

unset SAREMEM_DEMO_KEY
```

Add should return `success: true` and echo the three submitted IDs. Search returns a `data` array containing the original memory content, such as `[{"type":"text","text":"The bicycle is blue."}]`, with a generated memory ID and optional score/time fields. It returns evidence for answering the question. Repeating the unchanged Add request is idempotent.

## Method

```text
Add: ordered messages and inline images
  -> source captions and structured visual descriptions
  -> multiple textual views -> text embeddings
  -> atomic storage of original evidence and retrieval views

Search: query text / optional image / optional options
  -> initial seed candidates -> candidate relevance checks -> bounded evidence reordering
  -> validated evidence records from the requesting user's memory
```

The production Search adapter attempts the primary `simple` path, the `legacy` path, and then validated seed reuse or deterministic lexical seed retrieval. Successful empty results and short result lists do not trigger extra searches to fill `top_k`. Authentication, scope violations and source-integrity failures cannot be bypassed through recovery.

Each primary candidate assessment permits at most one transport retry and one protocol repair, with at most three actual sends sharing one stage deadline. The primary assessment client disables hidden SDK retries and redirects; the legacy client uses bounded retries. No whole-pipeline retry resets the request budget.

Add and Search use `gpt-4o-mini`; embeddings use `text-embedding-v4`. The implementation returns the original memory ID/content in selected rank order. It does not implement the organizer-owned answer-generation stage.

## Repository layout

```text
production/
  server.py                 # Production launcher; no automatic tests or ingestion
  init_db.py                # Explicit creation of a NEW database only
  configuration.py          # Explicit configuration reader; no shell execution
  aml_api/                  # API, storage, retrieval and production recovery
testing/
  testing_server.py         # Same app/service, separate test persistence/diagnostics
  run.py                    # offline | health | provider | smoke
  test_*.py                 # Offline synthetic and HTTP regression tests
LICENSE                    # MIT License
VERSION.json
INHERITED_SHA256.json       # Preserved component fingerprints
MANIFEST.json / SHA256SUMS  # Deterministic source-package identity and hashes
build_candidate.py         # Allowlisted distribution builder
```

Instance directories such as `config/`, `state/`, `testing/outputs/`, environments, caches, logs and backups are not part of the public distribution.

## Production deployment

Use a separate environment, service key and database for production. From a fresh checkout:

```sh
python3.12 -m venv .venv-prod
.venv-prod/bin/python -m pip install -r production/requirements.lock.txt
mkdir -p config state/production
chmod 700 config state/production
test -e config/production.env || cp production/config.env.example config/production.env
chmod 600 config/production.env
```

Edit `config/production.env` using the configuration reference below. Replace all placeholders and keep `AML_STORAGE_PATH=../state/production/memory.sqlite3`. Set host/port for your deployment; the template uses `127.0.0.1:8000`. Use a dedicated production credential and initialize only an absent database:

```sh
test -e state/production/memory.sqlite3 || \
  .venv-prod/bin/python -B production/init_db.py --path "$(pwd)/state/production/memory.sqlite3"
.venv-prod/bin/python -B production/server.py --config config/production.env
```

In a second terminal:

```sh
curl --fail http://127.0.0.1:8000/health
```

Expected: `{"status":"ok"}`. The production launcher uses one Uvicorn worker, limits CPU/BLAS threads, and emits no per-request diagnostics. Startup does not initialize a database or make model requests.

For persistent hosting, use an operator-managed supervisor or dedicated tmux session. Configure public HTTPS and the gateway deadline; use deployment-local caches, temporary paths and logs. Stop gracefully and allow child-worker cleanup. Preserve the production state directory and use SQLite's consistent backup mechanisms for a live database. Keep the synthetic smoke tool pointed at the test instance.

### Configuration reference

Edit configuration files on the deployment host. The reader accepts `AML_NAME=value` and separate comment lines; it does not remove surrounding quotes, execute shell syntax, or support inline comments. Storage paths resolve relative to the configuration file. Do not `source` these files into a shell.

| Setting                                           | Requirement                                                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `AML_SEARCH_PIPELINE`                             | Explicitly `simple` for this production launcher                                                          |
| `AML_MEMORY_SYSTEM_KEY`                           | A private participant-issued API credential, at least 24 characters; use different production/test values |
| `AML_LLM_API_KEY`, `AML_LLM_BASE_URL`             | LLM provider credentials and base URL                                                                     |
| `AML_EMBEDDING_API_KEY`, `AML_EMBEDDING_BASE_URL` | Embedding provider credentials and base URL                                                               |
| `AML_STORAGE_PATH`                                | A separate persistent path for each instance                                                              |
| `AML_BIND_HOST`, `AML_PORT`                       | Local listener; templates use loopback ports 8000/8001                                                    |
| `AML_TEST_BASE_URL`                               | URL of the isolated test instance used by local tools                                                     |
| `AML_PUBLIC_BASE_URL`                             | Deployment metadata; does not configure TLS or the listener                                               |

Replace every `CHANGE_ME` value. Never commit real configurations, keys, private provider URLs, databases or media. The **Memory System Key** protects this service; the **AML Eval/Leaderboard Key** belongs to the platform and is not a provider key or a server setting.

Small-host operators should explicitly set `AML_PROD_MAX_CONCURRENT_SEARCHES=1` and `AML_PROD_WORKER_MEMORY_MB=1024` in **both** instance configurations before validation. The package policy defaults are two searches and 2048 MiB of address space per worker. These limits are not memory reservations or proof of capacity. Configure CPU/BLAS threads conservatively, and declare supported Add/Search concurrency to AML based on measurement.

### Official AML evaluation

Register the public Add/Search endpoints, authentication, capacity and a fixed repository commit according to the [official participation and API instructions](https://agentmemoryleaderboard.ai/api-guide). AML sends inputs and runs **Add → Search → organizer Answer → organizer Eval**; the server does not need a preinstalled benchmark corpus for that flow. Gold answers, clues and categories must not be sent to or indexed by Add/Search.

Official Smoke is launched through AML after access/version binding; it is **not** `testing/run.py smoke`. It writes memory and can incur participant-side provider costs. Full follows the platform's Smoke and review gates. Do not change the evaluated code mid-run. Keep operational credentials out of repository URLs and public notes.

## API reference

`GET /health` is unauthenticated. `POST /add` and `POST /search` require the configured credential through `X-Api-Key`, `Authorization: Bearer ...`, or `Authorization: Token ...`.

This implementation accepts **ordered content-part arrays** for both message content and queries. A text-only request still uses a `[{"type":"text",...}]` array. Content parts are `text` or `image_url`; images must be inline Base64 Data URIs (JPEG/PNG/WebP), not remote image URLs or video blocks. Preserve message/content order. Supported decoded-image budgets are 10 MiB per image and 30 MiB per Add request or returned Search response.

Example Add request (schema illustration; sending it can incur charges):

```json
{
  "request_id": "example:write:1",
  "user_id": "example:user",
  "session_id": "example:session",
  "messages": [{
    "role": "user",
    "content": [{"type": "text", "text": "The bicycle is blue."}]
  }]
}
```

Success is `HTTP 200` **after persistence**, with `success: true` and unchanged `request_id`, `user_id` and `session_id`. Replays of the same logical write return the same acknowledgement; changing a payload under an existing request ID is a conflict.

Example Search request:

```json
{
  "query": [{"type": "text", "text": "What color is the bicycle?"}],
  "user_id": "example:user",
  "top_k": 100
}
```

Optional multiple-choice `options` is a top-level string array. Retrieval is scoped by the exact `user_id`; `session_id` organizes source sessions, not Search isolation. Success contains only `{"data":[...]}`. Each record has original `id`/ordered `content` and optional `score`/`created_at`. The return ceiling is `min(top_k, 20, configured return limit)`; fewer results or `data: []` can be valid. Search returns evidence, never a final answer, planner trace or provider payload.

### Errors and budgets

| Status    | Meaning                                                                        |
| --------- | ------------------------------------------------------------------------------ |
| 400 / 422 | Invalid media or schema; 422 contains safe field paths and correction messages |
| 401 / 403 | Invalid credentials / rejected memory scope                                    |
| 404 / 405 | Wrong endpoint / unsupported method                                            |
| 409       | Add conflict; an in-progress write is not a successful acknowledgement         |
| 429       | Search capacity exhausted; `Retry-After: 1`                                    |
| 500       | Internal failure or invalid success response                                   |
| 503 / 504 | Unavailable Search paths / exhausted Search deadline                           |

Errors are sanitized. Field diagnostics never echo values, unknown property names, images, arbitrary exception text or provider payloads. Organizer-side HTTP retries are distinct from the adapter's internal recovery. Follow the current [AML API guide](https://agentmemoryleaderboard.ai/api-guide) for external retry semantics and evaluation limits.

The application's Search budget defaults to, and cannot exceed, **1680 seconds**: primary 1140 s, legacy 420 s, seed reserve 90 s and cleanup 5 s per stage. Provider read timeout is 300 s and connect/write/pool limits are 30 s, clipped to remaining stage time on each send. An I/O read timeout is not an entire provider-call wall-clock limit. Request timing begins at the Search ASGI entry, while the stage supervisor bounds and reaps child processes.

The application deadline covers Search processing. Configure HTTPS and a gateway-wide deadline to also bound stalled uploads and downstream response transfers. Keep the gateway deadline below the platform limit and reserve time for transport.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Startup rejected | Replace placeholders, use a service key of at least 24 characters, verify provider URLs and explicitly initialize the configured database. |
| Configuration value rejected | Use plain `AML_NAME=value` lines without quotes or inline comments. Paths are relative to the configuration file. |
| Test launcher rejects a path | The database must be under `state/testing/`; diagnostics must use a new directory under `testing/outputs/`. |
| Health cannot connect | Keep terminal A running and match `AML_TEST_BASE_URL` to the listener host and port. |
| Provider probe fails | Check credentials, endpoint URLs, access to both required models, provider quota and network connectivity. |
| Smoke fails | Check Health and the provider probe first, then the test key/URL and the sanitized `requests.jsonl` stage diagnostics. |
| Offline TCP test cannot bind | Run on a host that permits local loopback listeners. |
| Search takes time | Real requests include remote model calls and bounded recovery; consult the Search budgets above and the test diagnostics. |

## Package verification

```sh
sha256sum -c SHA256SUMS
python3.12 -B build_candidate.py --output /absolute/path/to/a-new-saremem.tar.gz
```

Use `shasum -a 256 -c SHA256SUMS` where `sha256sum` is unavailable. The builder refuses an existing output archive, includes only allowlisted source/metadata, and validates component fingerprints. For Git checkouts, record `git rev-parse HEAD` to identify the repository revision; `MANIFEST.json` identifies the packaged file set, including documentation.
