/** Render a generator's `provider`/`method` pair as a single option label. */
export function genLabel(provider: string, method: string | null): string {
  return method ? `${provider}/${method}` : provider;
}
