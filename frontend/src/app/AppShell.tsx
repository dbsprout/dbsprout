import type { ReactNode } from "react";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div>
      <header>
        <h1>
          <span aria-hidden="true">🌱</span>{" "}
          <span>DBSprout Workbench</span>
        </h1>
      </header>
      <main>{children}</main>
    </div>
  );
}
