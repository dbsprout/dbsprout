import { afterEach, expect, test, vi } from "vitest";
import { connect, connectTest } from "./endpoints";
import type { SshTunnelInput } from "./types";

afterEach(() => vi.unstubAllGlobals());

function stubJson(body: unknown) {
  const m = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", m);
  return m;
}

function lastBody(m: ReturnType<typeof vi.fn>): unknown {
  const init = (m.mock.calls as unknown as [string, RequestInit][])[0][1];
  return JSON.parse(String(init.body));
}

test("connect POSTs { url } only when no ssh block is given", async () => {
  const m = stubJson({ source: "db", table_count: 1, tables: ["a"], dialect: "sqlite" });
  await connect("sqlite:///x.db");
  expect(lastBody(m)).toEqual({ url: "sqlite:///x.db" });
});

test("connect POSTs { url, ssh } when an ssh block is given", async () => {
  const m = stubJson({ source: "db", table_count: 1, tables: ["a"], dialect: "postgresql" });
  const ssh: SshTunnelInput = {
    host: "bastion.example.com",
    port: 22,
    user: "deploy",
    key_path: "/home/me/.ssh/id",
  };
  await connect("postgresql://u:p@db.internal:5432/app", ssh);
  expect(lastBody(m)).toEqual({ url: "postgresql://u:p@db.internal:5432/app", ssh });
});

test("connectTest POSTs { url } only when no ssh block is given", async () => {
  const m = stubJson({ ok: true, dialect: "sqlite", server_version: "3", table_count: 0, latency_ms: 1 });
  await connectTest("sqlite:///x.db");
  expect(lastBody(m)).toEqual({ url: "sqlite:///x.db" });
});

test("connectTest POSTs { url, ssh } when an ssh block is given", async () => {
  const m = stubJson({ ok: true, dialect: "postgresql", server_version: "16", table_count: 2, latency_ms: 5 });
  const ssh: SshTunnelInput = { host: "bastion", user: "deploy", key_path: "/k" };
  await connectTest("postgresql://u:p@db.internal:5432/app", ssh);
  expect(lastBody(m)).toEqual({ url: "postgresql://u:p@db.internal:5432/app", ssh });
});
