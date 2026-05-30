# DBSprout Workbench (frontend)

React + Vite + TypeScript single-page app served by FastAPI under `/app`.

## Develop
    npm ci
    npm run dev          # Vite dev server, proxies /api /ws /health to :8420
    # in another shell:  uv run dbsprout serve

## Build (emits into ../dbsprout/web/spa, served by `dbsprout serve`)
    npm run build

## Test / type-check
    npm test
    npm run typecheck
