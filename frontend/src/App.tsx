import { useEffect, useState } from "react";
import { fetchHealth } from "./api/health";

type Status = "checking" | "ok" | "error";

export function App() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    fetchHealth()
      .then((h) => setStatus(h.status === "ok" ? "ok" : "error"))
      .catch(() => setStatus("error"));
  }, []);

  return (
    <main>
      <h1>DBSprout Workbench</h1>
      <p data-testid="backend-status">backend: {status}</p>
    </main>
  );
}
