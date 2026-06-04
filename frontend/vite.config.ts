/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The built SPA is served by FastAPI under /app (assets under /app/assets), and
// emitted INTO the Python package so hatchling bundles it into the wheel.
export default defineConfig({
  base: "/app/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: fileURLToPath(new URL("../dbsprout/web/spa", import.meta.url)),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8420",
      "/health": "http://127.0.0.1:8420",
      "/ws": { target: "ws://127.0.0.1:8420", ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/setupTests.ts"],
    css: false,
    coverage: {
      provider: "v8",
      reporter: ["text", "json-summary"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/main.tsx",
        "src/**/*.test.{ts,tsx}",
        "src/test/**",
        "src/setupTests.ts",
        // Pure type-declaration module — no executable code to cover.
        "src/api/types.ts",
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
  },
});
