/**
 * Parse the seed input box into the wire value for POST /api/generate.
 *
 * The backend accepts `int | null` with `>= 0`; `null` means "server picks one".
 * A blank box → `null` (let the server materialise a seed). Any value that is
 * not a clean non-negative integer → `null` rather than a NaN/garbage number, so
 * the request body always satisfies the API contract.
 */
export function parseSeed(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  if (!/^\d+$/.test(trimmed)) return null;
  const n = Number(trimmed);
  return Number.isInteger(n) && n >= 0 ? n : null;
}
