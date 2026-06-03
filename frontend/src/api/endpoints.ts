import { ApiError, apiDownload, apiGet, apiPost, apiPut, apiUpload } from "./client";
import type {
  CancelJobResponse,
  ConnectionProbe,
  ConnectionsResponse,
  CostsResponse,
  DataSpec,
  DeleteConnectionResponse,
  ExportFormat,
  ExportRequest,
  GenerateRequest,
  GenerateResponse,
  GeneratorConfig,
  GeneratorsResponse,
  InsertPreview,
  InsertRequest,
  InsertResponse,
  JobRecordResponse,
  PreviewResponse,
  QualityResponse,
  RowCountResponse,
  RunsResponse,
  SamplesResponse,
  SavedConnectionInfo,
  SchemaSummary,
  SchemaTreeData,
  SpecAssistResponse,
  SpecProvider,
  TableAdvanced,
  TableAdvancedResponse,
  ValidateResponse,
} from "./types";

export const queryKeys = {
  samples: ["samples"] as const,
  schema: ["schema"] as const,
  spec: ["spec"] as const,
  generators: ["generators"] as const,
  preview: (table: string) => ["preview", table] as const,
  job: (jobId: string) => ["job", jobId] as const,
  validate: ["validate"] as const, // P1c-3
  runs: (page?: number) => ["runs", page] as const, // P1c-4
  quality: (runId?: number) => ["quality", runId] as const, // P1c-4
  costs: ["costs"] as const, // P1c-4
  connections: ["connections"] as const,
  // P2a-2
};

export const listSamples = () => apiGet<SamplesResponse>("/api/samples");
export const loadSample = (name: string) =>
  apiPost<SchemaSummary>("/api/schema/sample", { name });
export const getSchema = () => apiGet<SchemaTreeData>("/api/schema");

// Used by P1a-2b (connection form / paste); included now so the client is complete.
export const connect = (url: string) => apiPost<SchemaSummary>("/api/connect", { url });
export const connectTest = (url: string) =>
  apiPost<ConnectionProbe>("/api/connect/test", { url });
export const pasteSchema = (text: string, parser?: string) =>
  apiPost<SchemaSummary>("/api/schema/paste", { text, parser });

export const uploadSchema = (file: File, parser?: string) => {
  const form = new FormData();
  form.append("file", file);
  if (parser) form.append("parser", parser);
  return apiUpload<SchemaSummary>("/api/schema/load", form);
};

export const getSpec = () => apiGet<DataSpec>("/api/spec");
export const listGenerators = (dtype?: string) =>
  apiGet<GeneratorsResponse>(dtype ? `/api/generators?dtype=${encodeURIComponent(dtype)}` : "/api/generators");
export const putTableRowCount = (table: string, rowCount: number) =>
  apiPut<RowCountResponse>(`/api/spec/tables/${encodeURIComponent(table)}`, { row_count: rowCount });
export const putColumnSpec = (table: string, column: string, cfg: GeneratorConfig) =>
  apiPut<GeneratorConfig>(
    `/api/spec/tables/${encodeURIComponent(table)}/columns/${encodeURIComponent(column)}`,
    cfg,
  );
// ─── P2b-2: advanced packs (correlations + derived) ───
// Persist a table's correlations / derived lists; either may be omitted (partial
// update). Invalidate queryKeys.spec on success — no dedicated query key needed.
export const putTableAdvanced = (table: string, body: TableAdvanced) =>
  apiPut<TableAdvancedResponse>(
    `/api/spec/tables/${encodeURIComponent(table)}/advanced`,
    body,
  );
// ─── end P2b-2 ───
export const getPreview = (table: string) =>
  apiGet<PreviewResponse>(`/api/preview/${encodeURIComponent(table)}`);

// ── Generate (P1b-3) ─────────────────────────────────────────────────────
export const generate = (body: GenerateRequest) =>
  apiPost<GenerateResponse>("/api/generate", body);
export const getJob = (jobId: string) =>
  apiGet<JobRecordResponse>(`/api/jobs/${encodeURIComponent(jobId)}`);

// ─── P1c-4: insights ───
// Read-only JSON twins of the legacy HTML insights views, over state.db
// telemetry (backend dbsprout/web/routers/insights_api.py).
export const listRuns = (page?: number) =>
  apiGet<RunsResponse>(page === undefined ? "/api/runs" : `/api/runs?page=${page}`);
export const getQuality = (runId?: number) =>
  apiGet<QualityResponse>(runId === undefined ? "/api/quality" : `/api/quality?run_id=${runId}`);
export const getCosts = () => apiGet<CostsResponse>("/api/costs");

// ─── P1c-2: insert ─────────────────────────────────────────────────────────
// Write-guard preview → confirm → background insert job. Reuses getJob (P1b-3)
// + queryKeys.job for the poll — the job semantics are identical, so no new key.

/**
 * POST /api/insert/preview — request a single-use, scope-bound HMAC token for
 * the connected DB's last generation result. `tables` selects a subset (null /
 * omitted ⇒ whole-DB FK-safe scope).
 */
export const insertPreview = (tables?: string[]) =>
  apiPost<InsertPreview>("/api/insert/preview", { tables: tables ?? null });

/** POST /api/insert — start a background insert job (requires the preview token). */
export const insertData = (body: InsertRequest) =>
  apiPost<InsertResponse>("/api/insert", body);

/** POST /api/jobs/{id}/cancel — cooperatively cancel the active insert job. */
export const cancelJob = (jobId: string) =>
  apiPost<CancelJobResponse>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`);

// ─── P1c-1: export ───
// Imperative file download (not a query) — POST /api/export → blob → browser save.
export const exportData = (format: ExportFormat, tables?: string[]): Promise<void> => {
  const body: ExportRequest = tables && tables.length > 0 ? { format, tables } : { format };
  // The server names a single-table file "<table>.<ext>" and a multi-table bundle
  // "dbsprout-export.<ext>"; mirror that as the fallback when no header is present.
  const fallback =
    tables && tables.length === 1
      ? `${tables[0]}.${format}`
      : `dbsprout-export.${format}`;
  return apiDownload("/api/export", body, fallback);
};

// ─── P1c-3: validate ───
// POST /api/validate validates the last generation run (no body validates all
// tables; an optional table list scopes the report). The non-HTMX JSON branch
// is consumed here — no HX-Request header is sent.
export const validate = (tables?: string[]) =>
  apiPost<ValidateResponse>("/api/validate", tables ? { tables } : undefined);

// ─── P2a-2 ───
// Saved connections (.dbsprout/connections.toml). The server strips passwords
// before persisting; the client only ever sees/sends a password-stripped or
// ${ENV_VAR}-referenced URL.
export const getConnections = () => apiGet<ConnectionsResponse>("/api/connections");

export const saveConnection = (name: string, url: string) =>
  apiPost<SavedConnectionInfo>("/api/connections", { name, url });

// DELETE has no client.ts helper (that module is owned elsewhere), so this issues
// the request inline and reuses ApiError for the typed-envelope error path.
export const deleteConnection = async (name: string): Promise<DeleteConnectionResponse> => {
  const resp = await fetch(`/api/connections/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!resp.ok) {
    let detail: { code?: string; message?: string; hint?: string } = {};
    try {
      const data = (await resp.json()) as { detail?: typeof detail } | null;
      detail = data?.detail ?? {};
    } catch {
      // Non-JSON error body — fall through to a status-only ApiError.
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as DeleteConnectionResponse;
};

// ─── P2b-3 ───
// POST /api/spec/assist — let an LLM propose a full DataSpec for the loaded
// schema. Default provider is the offline "embedded" path; "cloud" is opt-in.
// On success the server stores the proposal on the workspace, so callers
// invalidate queryKeys.spec to repaint the configure grid (no new query key).
export const assistSpec = (provider?: SpecProvider) =>
  apiPost<SpecAssistResponse>("/api/spec/assist", provider ? { provider } : {});
