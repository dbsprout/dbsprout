import { apiGet, apiPost, apiUpload } from "./client";
import type {
  ConnectionProbe,
  SamplesResponse,
  SchemaSummary,
  SchemaTreeData,
} from "./types";

export const queryKeys = {
  samples: ["samples"] as const,
  schema: ["schema"] as const,
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
