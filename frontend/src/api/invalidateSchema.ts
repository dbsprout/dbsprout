import type { QueryClient } from "@tanstack/react-query";
import { queryKeys } from "./endpoints";

// ─── P5-12 ───
// Loading a new schema changes more than the schema tree: the `spec` (configure
// grid + wizard `hasSpec` gate) and every per-table `preview` are derived from
// it. The loaders previously invalidated only `queryKeys.schema`, so a `spec`
// query left in its initial 409 ("No schema loaded") error state never refetched
// — Configure stayed stuck and Next stayed disabled until a hard page refresh.
//
// This helper is the single place that fans a schema-load invalidation out to
// all three families. `preview` is keyed `["preview", <table>]`, so we invalidate
// the `["preview"]` PREFIX (TanStack Query's default non-exact match) to refresh
// every table's preview regardless of which one is active.
export function invalidateSchemaQueries(qc: QueryClient): void {
  qc.invalidateQueries({ queryKey: queryKeys.schema });
  qc.invalidateQueries({ queryKey: queryKeys.spec });
  qc.invalidateQueries({ queryKey: ["preview"] });
}
