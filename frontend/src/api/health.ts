export interface Health {
  status: string;
}

export async function fetchHealth(): Promise<Health> {
  const resp = await fetch("/health");
  if (!resp.ok) {
    throw new Error(`health check failed: ${resp.status}`);
  }
  return (await resp.json()) as Health;
}
