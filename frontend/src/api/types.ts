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

// ── P1c-3: Validate ──
// Mirrors POST /api/validate's JSON (non-HTMX) envelope — see
// dbsprout/web/routers/validate.py (_aggregate_report / _serialise_fidelity /
// _serialise_detection). Keys `fidelity` and `detection` are always present but
// null when no reference rows are seeded or the optional [stats] extra is absent.

/** Optional list of table names to scope validation to (server validates the last run). */
export interface ValidateRequest {
  tables?: string[];
}

/** High-level integrity totals across the validated run. */
export interface ValidateSummary {
  tables: number;
  rows: number;
  /** Count of failed checks (not bad rows). */
  violations: number;
}

/** Per-table violation buckets; one row per schema table (zeros when clean). */
export interface IntegrityByTable {
  table: string;
  fk_violations: number;
  unique_violations: number;
  not_null_violations: number;
  check_violations: number;
}

/** A single failed integrity check (details list is capped at 500 server-side). */
export interface IntegrityDetail {
  check: string;
  table: string;
  /** Null for table-level checks (e.g. composite PK). */
  column: string | null;
  /** Always false in details (passing checks are not listed). */
  passed: boolean;
  details: string;
}

/** One fidelity distribution-similarity metric. */
export interface FidelityMetric {
  metric: string;
  table: string;
  column: string | null;
  score: number;
  details: string;
}

export interface FidelityReport {
  overall_score: number;
  passed: boolean;
  metrics: FidelityMetric[];
}

/** One detection (C2ST) metric. */
export interface DetectionMetric {
  metric: string;
  table: string;
  accuracy: number;
  details: string;
}

export interface DetectionReport {
  overall_score: number;
  passed: boolean;
  metrics: DetectionMetric[];
}

export interface ValidateResponse {
  summary: ValidateSummary;
  by_table: IntegrityByTable[];
  details: IntegrityDetail[];
  fidelity: FidelityReport | null;
  detection: DetectionReport | null;
}
