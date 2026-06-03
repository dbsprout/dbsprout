import { apiGet, apiPost, apiPut, apiUpload } from "./client";
import type {
  ConnectionProbe,
  CostsResponse,
  DataSpec,
  GenerateRequest,
  GenerateResponse,
  GeneratorConfig,
  GeneratorsResponse,
  JobRecordResponse,
  PreviewResponse,
  QualityResponse,
  RowCountResponse,
  RunsResponse,
  SamplesResponse,
  SchemaSummary,
  SchemaTreeData,
} from "./types";

export const queryKeys = {
  samples: ["samples"] as const,
  schema: ["schema"] as const,
  spec: ["spec"] as const,
  generators: ["generators"] as const,
  preview: (table: string) => ["preview", table] as const,
  job: (jobId: string) => ["job", jobId] as const,
  runs: (page?: number) => ["runs", page] as const, // P1c-4
  quality: (runId?: number) => ["quality", runId] as const, // P1c-4
  costs: ["costs"] as const, // P1c-4
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
