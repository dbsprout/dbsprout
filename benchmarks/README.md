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

The `benchmark` job in `.github/workflows/ci.yml` runs the suite
non-blocking (`continue-on-error: true`) and uploads
`benchmark-results.json` as a build artifact.

> Historical baseline comparison with a >20% regression gate requires a
> persistent artifact store (e.g. gh-pages or an external benchmark store)
> and is tracked as a follow-up story.
