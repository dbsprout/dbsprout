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

// ── P1c-2: Insert ──────────────────────────────────────────────────────────
// Mirrors POST /api/insert/preview + POST /api/insert + POST /api/jobs/{id}/cancel
// (backend dbsprout/web/routers/insert.py + jobs.py).

/** Writer strategy pinned on the insert request (S-141 method select). */
export type InsertMethod = "auto" | "batch" | "copy";

/** One table in the FK-safe insert scope, with its row count. */
export interface InsertScopeEntry {
  table: string;
  row_count: number;
}

/**
 * Response of POST /api/insert/preview — issues a single-use, scope-bound HMAC
 * confirmation_token (5-min TTL). The token carries only hashes server-side
 * (never the raw DSN); the redacted target + dialect + scope are safe to render.
 * `warnings` is optional: the current backend preview does not emit it, but the
 * client tolerates a future preview that does (FK-prerequisite hints also arrive
 * on the insert response's `scope_warnings`).
 */
export interface InsertPreview {
  target: string;
  dialect: string;
  scope: InsertScopeEntry[];
  total_rows: number;
  confirmation_token: string;
  warnings?: string[];
}

/** Request body for POST /api/insert. The token comes from a prior preview. */
export interface InsertRequest {
  tables?: string[] | null;
  confirmation_token: string;
  method: InsertMethod;
}

/** Response of POST /api/insert — starts a background job. */
export interface InsertResponse {
  job_id: string;
  scope: InsertScopeEntry[];
  total_rows: number;
  writer: string;
  method: InsertMethod;
  scope_warnings: string[];
}

/** Response of POST /api/jobs/{id}/cancel. */
export interface CancelJobResponse {
  job_id: string;
  status: string;
}
