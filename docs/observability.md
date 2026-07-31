# Loki + Grafana Observability Guide (Colab)

This guide sets up live log observability for the benchmark suite so you
can watch exactly what each phase is doing in real time — per-batch
progress, per-request TTFT/tokens-per-second, GPU utilization, and
accuracy eval progress — instead of staring at a silent Colab cell for
several minutes.

Docker is intentionally **not** used here since Colab does not reliably
support nested/privileged Docker daemons. Instead, Loki, Promtail, and
Grafana run as standalone binaries directly inside the Colab VM.

## 1. Start the stack

Run this once, right after Phase 0.6 (it's already wired in as **Phase
0.7** in the notebook):

```python
!bash observability/setup_observability.sh
```

This downloads (first run only, cached after) and starts three background
processes:

- **Loki** (port 3100) — the log storage/query backend.
- **Promtail** (port 9080) — tails `logs/*.log` and ships new lines to Loki.
- **Grafana** (port 3000) — the dashboard UI you'll actually look at.

## 2. Open Grafana from Colab

Colab blocks direct `localhost` browser access, so use Colab's built-in
port proxy (already included in the Phase 0.7 cell):

```python
from google.colab.output import eval_js
grafana_url = eval_js("google.colab.kernel.proxyPort(3000)")
print(grafana_url)
```

Click the printed URL — it opens Grafana with anonymous admin access
already enabled (no login needed, configured via
`GF_AUTH_ANONYMOUS_ENABLED` in `setup_observability.sh`).

## 3. Add Loki as a data source (one-time setup)

1. In Grafana, go to **Connections → Data sources → Add data source**.
2. Choose **Loki**.
3. Set the URL to `http://localhost:3100`.
4. Click **Save & test** — you should see a green success message.

## 4. Build a live log panel

1. Go to **Dashboards → New → New Dashboard → Add visualization**.
2. Select the Loki data source.
3. In the query field, use LogQL to filter by technique/phase, e.g.:
   ```
   {job="llm_bench"} | json | technique="fp16"
   ```
4. Switch the visualization type to **Logs** for a scrolling live feed,
   or **Time series** if you want to graph a numeric field like
   `running_avg_tps` or `tokens_per_second` over time (Loki can extract
   numeric fields from the JSON payload for graphing — use the
   **unwrap** LogQL operator, e.g. `{job="llm_bench"} | json | unwrap tokens_per_second`).
5. Set the dashboard's auto-refresh (top-right) to **5s** so it updates
   live while a phase cell is running.

## 5. What to watch during each phase

Every log line from `benchmark_runner.py` includes structured fields you
can filter/graph on in Grafana:

| Field | Meaning |
|---|---|
| `technique` | `fp16`, `gptq`, `awq`, `gguf_q4_k_m` |
| `phase` | `start`, `batch_start`, `warmup`, `warmup_done`, `measuring`, `batch_done`, `finished` |
| `batch_size` | current batch size being tested |
| `step` / `total` | request N of M within the current batch/warmup |
| `ttft_s` | time-to-first-token for that single request |
| `tokens_per_second` | throughput for that single request |
| `running_avg_tps` | running average throughput across the current batch so far |

A useful live query while Phase A-E cells are running:

```
{job="llm_bench"} | json | phase="measuring"
```

This shows a live-updating feed of every request's TTFT and throughput as
it happens, so you can see exactly where time is going instead of a
multi-minute silent gap.

## 6. Plain-text fallback (no Grafana needed)

If you just want readable logs directly in the Colab cell without
Grafana, set `PLAIN_LOGS=1` before running a phase cell:

```python
import os
os.environ["PLAIN_LOGS"] = "1"
```

This switches `src/log.py` to emit human-readable lines instead of JSON,
while still writing to `logs/*.log` for Promtail to pick up if you decide
to check Grafana later.

## 7. Stopping the stack

At the end of your session (or before restarting the Colab runtime):

```bash
!bash observability/stop_observability.sh
```

## Troubleshooting

- **Grafana proxy URL shows a blank page**: wait ~10s after running Phase
  0.7 before opening the URL — Grafana takes a few seconds to bind its port.
- **No logs appearing in Grafana**: confirm `logs/*.log` files exist
  (`!ls -la logs/`) and that Promtail started without errors
  (`!cat logs/promtail.out`).
- **Loki data source test fails**: confirm Loki is running with
  `!curl -s http://localhost:3100/ready` — it should return `ready`.
