import { apiDownload, apiGet, apiPost, apiPut, apiUpload } from "./client";
import type {
  ConnectionProbe,
  DataSpec,
  ExportFormat,
  ExportRequest,
  GenerateRequest,
  GenerateResponse,
  GeneratorConfig,
  GeneratorsResponse,
  JobRecordResponse,
  PreviewResponse,
  RowCountResponse,
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
