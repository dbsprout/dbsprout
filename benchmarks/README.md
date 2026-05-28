# Performance Benchmarks (S-074)

Performance benchmark suite. **Not** part of the default `pytest` run
(`testpaths = ["tests"]`) and **not** counted toward the 95% coverage gate
(`coverage source = ["dbsprout"]`). Benchmark files are named `bench_*.py`;
they live here and are only collected via an explicit `pytest benchmarks/`
invocation.

## Run locally

```bash
uv run pytest benchmarks/
```

This runs every timed benchmark **and** the memory-ceiling assertion.
Add `--benchmark-only` to skip the non-benchmark memory test:

```bash
uv run pytest benchmarks/ --benchmark-only
```

## Machine-readable results (trend analysis)

```bash
uv run pytest benchmarks/ --benchmark-json=benchmark-results.json
```

The JSON artifact contains per-benchmark min/max/mean/median/stddev/OPS and
the machine's CPU info — suitable for tracking performance trends across
releases.

## Benchmarks

| Test | What it measures | Perf target (CLAUDE.md) |
|------|------------------|--------------------------|
| `test_bench_heuristic_generation` | rows/sec, heuristic engine | 100K+ rows/sec |
| `test_bench_numpy_vectorized` | values/sec, NumPy fast path | high-throughput numeric |
| `test_bench_fk_sampling` | FK sample speed (10k parents) | 100% FK integrity |
| `test_bench_introspection` | SQLite schema introspection | — |
| `test_bench_sql_output` | SQL INSERT writer throughput | — |
| `test_bench_generation_memory` | peak memory, 100k-row pass | mem < 2 GB |
| `test_bench_cli_startup` | `dbsprout --help` cold start | CLI startup < 500ms |

## CI

Two jobs in `.github/workflows/ci.yml` consume this suite:

- **`benchmark`** — runs the suite non-blocking (`continue-on-error: true`)
  and uploads `benchmark-results.json` as a build artifact (trend analysis).
- **`benchmark-gate`** — the **blocking** regression gate (S-096). It runs
  the suite and then compares the result against the committed baseline,
  failing CI on a regression.

## Regression gate (S-096)

`check_regression.py` compares a fresh `benchmark-results.json` against the
committed baseline `baseline.json` and **fails when any benchmark's mean run
time regresses by more than 20%** (a benchmark regresses when
`current.mean > baseline.mean * 1.20`). The 20% margin is wide enough to
tolerate normal CI hardware variance while still catching real regressions.

Run it locally:

```bash
uv run pytest benchmarks/ --benchmark-json=benchmark-results.json
uv run python benchmarks/check_regression.py \
    --baseline benchmarks/baseline.json \
    --current benchmark-results.json
```

Exit codes: `0` (no regression), `1` (regression found — CI fails), `2`
(bad input). A custom threshold can be passed with `--threshold 0.15`.

Matching is by benchmark `fullname`. Benchmarks present in only one of the two
files are ignored (a new benchmark cannot regress; a removed one is not a
regression), so adding or removing a benchmark never breaks the gate.

### Refreshing the baseline

The committed baseline (`benchmarks/baseline.json`) is a slim file holding only
each benchmark's `fullname` and `stats.mean`. Refresh it after an *intentional*
performance change, when merging to `dev`/`main`:

```bash
uv run pytest benchmarks/ --benchmark-json=benchmark-results.json
python - <<'PY'
import json
raw = json.load(open("benchmark-results.json"))
slim = {
    "version": raw.get("version"),
    "benchmarks": [
        {"name": b["name"], "fullname": b["fullname"], "stats": {"mean": b["stats"]["mean"]}}
        for b in raw["benchmarks"]
    ],
}
json.dump(slim, open("benchmarks/baseline.json", "w"), indent=2)
open("benchmarks/baseline.json", "a").write("\n")
PY
git add benchmarks/baseline.json
```

> A future story may automate baseline persistence on merge (e.g. via
> `github-action-benchmark` + gh-pages) to remove the manual refresh step.
