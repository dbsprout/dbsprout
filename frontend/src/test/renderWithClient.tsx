import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { SelectionProvider } from "../app/SelectionProvider";

export function renderWithClient(ui: ReactElement): RenderResult {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  // SelectionProvider is an in-memory, null-default context (P4-7). Wrapping it
  // here lets panels that read the cross-panel selection store render bare in
  // tests without each suite re-declaring the provider; it changes no behaviour
  // until a `setSelection` is dispatched.
  return render(
    <QueryClientProvider client={client}>
      <SelectionProvider>{ui}</SelectionProvider>
    </QueryClientProvider>,
  );
}
