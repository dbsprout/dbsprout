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
