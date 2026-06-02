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
