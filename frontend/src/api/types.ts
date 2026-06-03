// Mirrors the DBSprout web JSON API (backend DBS-182).

export interface ApiErrorBody {
  code: string;
  message: string;
  correlation_id?: string;
  hint?: string;
}

export interface SampleInfo {
  name: string;
  title: string;
  description: string;
  dialect: string;
  table_count: number;
}

export interface SamplesResponse {
  samples: SampleInfo[];
}

export interface SchemaSummary {
  source: string;
  table_count: number;
  tables: string[];
  dialect: string;
}

export interface ConnectionProbe {
  ok: boolean;
  dialect: string;
  server_version: string;
  table_count: number;
  latency_ms: number;
}

export interface ColumnNode {
  name: string;
  type: string;
  nullable: boolean;
  unique: boolean;
  autoincrement: boolean;
  default: string | null;
  max_length: number | null;
}

export interface ForeignKeyNode {
  columns: string[];
  ref_table: string;
  ref_columns: string[];
  on_delete: string | null;
}

export interface TableNode {
  name: string;
  primary_key: string[];
  columns: ColumnNode[];
  foreign_keys: ForeignKeyNode[];
}

export interface SchemaTreeData {
  table_count: number;
  dialect: string | null;
  source: string | null;
  tables: TableNode[];
}

export interface GeneratorConfig {
  provider: string;
  method: string | null;
  params: Record<string, unknown>;
  distribution: string | null;
  distribution_params: Record<string, number>;
  min_value: number | null;
  max_value: number | null;
  enum_values: string[] | null;
  format_pattern: string | null;
  unique: boolean;
  nullable_rate: number;
  vectorized: boolean;
}

export interface TableSpec {
  table_name: string;
  row_count: number;
  columns: Record<string, GeneratorConfig>;
  derived: unknown[];
  correlations: unknown[];
  cardinality: Record<string, unknown> | null;
}

export interface DataSpec {
  version: string;
  tables: TableSpec[];
  global_seed: number;
  schema_hash: string;
  model_used: string | null;
  created_at: string | null;
}

export interface GeneratorMethod {
  provider: string;
  method: string;
  description: string;
  example: string;
  dtypes: string[];
  params: string[];
}

export interface GeneratorsResponse {
  providers: string[];
  methods: GeneratorMethod[];
}

export interface RowCountResponse {
  table_name: string;
  row_count: number;
}

export interface PreviewResponse {
  table: string;
  limit: number;
  total: number;
  rows: Record<string, unknown>[];
}

// ── Generate (P1b-3) ─────────────────────────────────────────────────────
// Mirrors POST /api/generate + GET /api/jobs/{id} (backend dbsprout/web/routers/generate.py).

/** The four registered generation engines. */
export type Engine = "heuristic" | "spec" | "statistical" | "finetuned";

export interface GenerateRequest {
  engine: Engine;
  // null / omitted → the server materialises a fresh non-negative 63-bit seed.
  seed: number | null;
}

export interface GenerateResponse {
  job_id: string;
  // The seed actually used (echoed back, materialised server-side when sent null).
  seed: number;
}

/** Terminal states are succeeded / failed / cancelled; running is the only live state. */
export type JobStatus = "running" | "succeeded" | "failed" | "cancelled";

export interface JobRecordResponse {
  id: string;
  kind: string;
  status: JobStatus;
  engine: string | null;
  seed: number | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

// ── P1c-4: Runs/Quality/Costs ──
// Mirrors GET /api/runs · /api/quality · /api/costs
// (backend dbsprout/web/routers/insights_api.py). Telemetry only — no secrets.

export interface RunRow {
  id: number | null;
  started_at: string;
  engine: string;
  provider: string | null;
  total_rows: number;
  total_tables: number;
  duration_ms: number | null;
  cost: number;
}

export interface RunsResponse {
  rows: RunRow[];
  page: number;
  total_pages: number;
  total_runs: number;
  has_prev: boolean;
  has_next: boolean;
}

/** Classified quality-metric status. */
export type QualityStatus = "pass" | "fail" | "warn";

export interface QualityRow {
  metric_type: string;
  metric_name: string;
  score: number;
  passed: boolean;
  status: QualityStatus;
  details_json: string | null;
}

export interface QualityResponse {
  // false (with empty rows, run_id null) when no run exists / run_id unknown.
  found: boolean;
  run_id: number | null;
  rows: QualityRow[];
}

export interface ProviderCost {
  provider: string;
  cost: number;
  tokens: number;
  calls: number;
}

export interface CostsResponse {
  total_cost: number;
  total_tokens: number;
  total_calls: number;
  avg_cost_per_run: number;
  per_provider: ProviderCost[];
}
