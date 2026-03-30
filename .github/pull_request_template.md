<!-- All PRs must target the `dev` branch. Never PR directly to `main`. -->

## Summary

<!-- 1-3 bullet points describing what this PR does and why -->

-

## Related Issues

<!-- Link Linear issues: DBS-XX -->

Closes DBS-

## Changes

<!-- List key changes grouped by area -->

### Added
-

### Changed
-

## Quality Checklist

### Tests (required)

- [ ] Unit tests written for all new/changed code
- [ ] All tests passing: `uv run pytest`
- [ ] Coverage **>= 95%**: `uv run pytest --cov=dbsprout --cov-fail-under=95`
- [ ] Edge cases and error paths tested

### Code Quality (required)

- [ ] Linting clean: `uv run ruff check .`
- [ ] Formatting clean: `uv run ruff format --check .`
- [ ] Type checking clean: `uv run mypy --strict dbsprout/`
- [ ] Security scan clean: `uv run bandit -c pyproject.toml -r dbsprout/`

### Design (required)

- [ ] No hardcoded secrets or credentials
- [ ] Immutable data patterns used (no mutation of existing objects)
- [ ] Functions < 50 lines, files < 800 lines
- [ ] Errors handled explicitly, not silently swallowed
- [ ] All user/external input validated at boundaries

### Documentation

- [ ] Public API docstrings added/updated
- [ ] Acceptance criteria from story validated
- [ ] Breaking changes documented (if any)

## Test Plan

<!-- How to verify this PR works. Include specific commands or scenarios. -->

```bash
uv run pytest tests/path/to/relevant_tests.py -v
```

## Screenshots / Output

<!-- If applicable: terminal output, dashboard screenshots, benchmark results -->
