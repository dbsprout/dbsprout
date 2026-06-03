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

// ─── P2b-2: advanced packs (correlations + derived) ───
// Mirrors dbsprout.spec.models.CorrelationRule / DerivedColumn. Persisted via
// PUT /api/spec/tables/{t}/advanced (backend dbsprout/web/routers/spec.py).

/** A multi-column coherence rule (e.g. city/state/zip lookup, FK fan-out). */
export interface CorrelationRule {
  columns: string[];
  lookup_table: string | null;
  strategy: string;
}

/** An expression-based column derived from other columns on the same table. */
export interface DerivedColumn {
  column: string;
  expression: string;
  depends_on: string[];
}

/** Request body for the advanced PUT — either list may be omitted (partial update). */
export interface TableAdvanced {
  correlations?: CorrelationRule[];
  derived?: DerivedColumn[];
}

/** Response of PUT /api/spec/tables/{t}/advanced. */
export interface TableAdvancedResponse {
  table_name: string;
  correlations: CorrelationRule[];
  derived: DerivedColumn[];
}
// ─── end P2b-2 ───

export interface TableSpec {
  table_name: string;
  row_count: number;
  columns: Record<string, GeneratorConfig>;
  // ─── P2b-2 ─── (narrowed from unknown[] to the typed advanced-pack arrays)
  derived: DerivedColumn[];
  correlations: CorrelationRule[];
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

// ─── P4-4 ───
// Mirrors GET /api/jobs/{id}/result (backend dbsprout/web/routers/generate.py).
// The richer, terminal-only result envelope: the *real* per-table generated row
// counts + per-table / total duration, read off the GenerateResult captured on
// the JobRecord (vs. the /api/spec approximation). Available only once a run
// succeeds (the endpoint 409s before then, 404s on an unknown id).

/** One generated table's actual row count + real generation time (ms). */
export interface JobTableResult {
  table_name: string;
  row_count: number;
  duration_ms: number;
}

/** Response of GET /api/jobs/{id}/result — actual generated counts + timings. */
export interface JobResultResponse {
  job_id: string;
  total_rows: number;
  total_tables: number;
  total_duration_ms: number;
  tables: JobTableResult[];
}
// ─── end P4-4 ───

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

// ── P1c-1: Export ──
// Mirrors POST /api/export (backend dbsprout/web/routers/export.py).

/** The four file-format writers offered for export. */
export type ExportFormat = "sql" | "csv" | "json" | "parquet";

export interface ExportRequest {
  format: ExportFormat;
  // Omitted / undefined → every table in the last run (FK-safe order preserved).
  tables?: string[];
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

// ─── P2b-3 ───
// Mirrors POST /api/spec/assist (backend dbsprout/web/routers/spec_assist.py).
// An LLM proposes a full DataSpec for the loaded schema; the server stores it on
// the workspace (GET /api/spec then reflects it) and returns this summary.

/** Which provider proposes the spec. `embedded` is offline; `cloud` is opt-in. */
export type SpecProvider = "embedded" | "cloud";

/** Summary returned by POST /api/spec/assist after a successful proposal. */
export interface SpecAssistResponse {
  provider: SpecProvider;
  model_used: string | null;
  schema_hash: string;
  tables: number;
  total_columns: number;
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

// ─── P2a-2 ───
// Saved connections persisted to .dbsprout/connections.toml
// (backend dbsprout/web/routers/connections.py). Passwords are NEVER stored —
// `url` is always password-stripped (empty or an ${ENV_VAR} reference).

/** One saved connection — a name and a password-stripped URL. */
export interface SavedConnectionInfo {
  name: string;
  url: string;
}

/** Response of GET /api/connections. */
export interface ConnectionsResponse {
  connections: SavedConnectionInfo[];
}

/** Request body for POST /api/connections. */
export interface SaveConnectionRequest {
  name: string;
  url: string;
}

/** Response of DELETE /api/connections/{name}. */
export interface DeleteConnectionResponse {
  deleted: boolean;
}

// ─── P2a-3 ───
// Optional SSH bastion block carried alongside the URL on POST /api/connect and
// /api/connect/test (backend dbsprout/core/ssh_tunnel.py + routers/connect.py).
// The private key is referenced by PATH only — its bytes are never uploaded.
// When omitted, the connect request is a plain `{ url }` (behaviour unchanged).

/** SSH bastion descriptor sent to tunnel a DB connection. */
export interface SshTunnelInput {
  /** Bastion (jump) host. */
  host: string;
  /** Bastion SSH port (defaults to 22 server-side when omitted). */
  port?: number;
  /** SSH username on the bastion. */
  user: string;
  /** Path to the SSH private key (referenced by path; never uploaded). */
  key_path: string;
}
