/**
 * Plain-language explanation + remedy for a failed integrity check. The backend's
 * `details[].check` keys are not 1:1 with the four integrity buckets (e.g.
 * `pk_uniqueness` and `unique` both mean a duplicate key), so we classify by a
 * case-insensitive substring match rather than an exhaustive enum — robust to
 * backend variance while still giving a precise message for the known checks.
 */
export interface ViolationHelp {
  /** What the violation means, in one sentence a non-expert can read. */
  what: string;
  /** How to fix it — concrete, actionable, points at Generate or Configure. */
  fix: string;
}

type Kind = "duplicate" | "fk" | "not_null" | "check" | "unknown";

/**
 * Classify a backend check key into a help kind. Matching is case-insensitive and
 * token-aware: the key is split on non-alphanumerics so `check_constraint` matches
 * but an unrelated `mystery_check`-style key only matches if `check` is a real
 * token (it is) — so we additionally anchor "check" to the known constraint forms
 * to avoid mis-classifying arbitrary `*_check` names. The earlier, more specific
 * kinds win first.
 */
function classify(check: string): Kind {
  const c = check.toLowerCase();
  const tokens = c.split(/[^a-z0-9]+/).filter(Boolean);
  const has = (t: string) => tokens.includes(t);
  // `pk_uniqueness` and `unique` both mean "duplicate key value(s)".
  if (has("pk") || has("unique") || c.includes("uniqueness")) return "duplicate";
  if (has("fk") || has("foreign")) return "fk";
  if (c.includes("not_null") || c.includes("notnull")) return "not_null";
  // Only the dedicated CHECK-constraint check, not any key that happens to end
  // in `_check`, maps to the CHECK remedy.
  if (has("check") && (tokens.length === 1 || has("constraint") || has("violations") || has("satisfaction"))) {
    return "check";
  }
  return "unknown";
}

const HELP: Record<Kind, ViolationHelp> = {
  duplicate: {
    what: "Two or more rows share the same value(s) for a key that must be unique.",
    fix: "Re-generate — a different seed usually clears the collision — or lower the row count for this table. Composite primary keys built from foreign keys are de-duplicated automatically.",
  },
  fk: {
    what: "A row references a parent row that does not exist (an orphaned foreign key).",
    fix: "Re-generate so the reference is sampled from real parent rows, or generate the parent table first / increase its row count.",
  },
  not_null: {
    what: "A column that must always have a value contains a NULL.",
    fix: "Set this column's nullable rate to 0 in Configure, or pick a generator that always produces a value.",
  },
  check: {
    what: "A value violates a CHECK constraint (an allowed-range or allowed-set rule on the column).",
    fix: "Adjust the generator's min/max or allowed values in Configure so every value satisfies the constraint.",
  },
  unknown: {
    what: "This integrity check failed for the generated data.",
    fix: "Re-generate with a different seed, or adjust this column's generator in Configure.",
  },
};

/** Map a check key to its plain-language explanation + remedy. */
export function violationHelp(check: string): ViolationHelp {
  return HELP[classify(check)];
}

const REASON: Record<Kind, string> = {
  duplicate: "duplicate key value — re-generate (a new seed usually clears it)",
  fk: "orphaned foreign key — re-generate to resample parent rows",
  not_null: "NULL in a non-nullable column — set this column's nullable rate to 0",
  check: "value violates a CHECK constraint — adjust min/max or allowed values",
  unknown: "flagged by a failed integrity check — re-generate or adjust the generator",
};

/** A short one-liner used as the cross-panel drill reason shown in Configure. */
export function drillReason(check: string): string {
  return REASON[classify(check)];
}
