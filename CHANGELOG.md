# Changelog

## [0.2.0](https://github.com/dbsprout/dbsprout/compare/v0.1.8...v0.2.0) (2026-06-04)


### Features

* **core:** add probe_connection() primitive (P1a) ([a87efc3](https://github.com/dbsprout/dbsprout/commit/a87efc3da0e61179173aa9f81c11a5fc2452d33b))
* **core:** bundle sample schemas + registry (P1a) ([a96ba23](https://github.com/dbsprout/dbsprout/commit/a96ba230e68fbf802a8bd4240376e1848f3ab08b))
* **S-152:** bump GitHub Actions to Node24-compatible ([bbba5be](https://github.com/dbsprout/dbsprout/commit/bbba5be9a3ff596e5c61f6529dfb09a492064fc2))
* **S-152:** bump GitHub Actions to Node24-compatible ([0c29b4a](https://github.com/dbsprout/dbsprout/commit/0c29b4a722ddd9c42c92a4d9cde02c72a9b06662))
* **web:** 7-step model with cache-reading gate predicates (P3-1) ([0cc5cc1](https://github.com/dbsprout/dbsprout/commit/0cc5cc1f27936693727d772b65fc4cca36f43a7b))
* **web:** add ColumnGrid with generator select (P1b-2) ([bdc8a36](https://github.com/dbsprout/dbsprout/commit/bdc8a363ac4ba2d467570f3f31f01b749f0d7da8))
* **web:** add ColumnInspector side panel (P1b-2) ([ccbbc51](https://github.com/dbsprout/dbsprout/commit/ccbbc514558cb3132ad0adf3b86c1f1c6a92509a))
* **web:** add ConfigurePanel composing grid+inspector+preview with live sample (P1b-2) ([bf6e196](https://github.com/dbsprout/dbsprout/commit/bf6e1969e758c5228006f0bc5d7218a18665630f))
* **web:** add genLabel helper + PreviewTable component (P1b-2) ([8465dd1](https://github.com/dbsprout/dbsprout/commit/8465dd1692e3c4d8d80164da0d9fda9db75f51ab))
* **web:** add PreviewResponse type + getPreview client fn (P1b-2) ([529be02](https://github.com/dbsprout/dbsprout/commit/529be023031c7b9c56ce583d99cd08b10f10d697))
* **web:** Advanced packs — AI spec-assist (P2b-3) ([b46e425](https://github.com/dbsprout/dbsprout/commit/b46e425b2f7f3ebdea7fbca8d68d55f1b3105e13))
* **web:** Advanced packs — AI spec-assist (P2b-3) ([6c84a26](https://github.com/dbsprout/dbsprout/commit/6c84a263c268f194bc501435e5d46ec027f81ee4))
* **web:** Advanced packs — correlations + derived columns (P2b-2) ([bf7c7c1](https://github.com/dbsprout/dbsprout/commit/bf7c7c1dea42eff951bee5a48f8f997405cbdaf6))
* **web:** Advanced packs — correlations + derived columns (P2b-2) ([7714d46](https://github.com/dbsprout/dbsprout/commit/7714d46e5e4afa800065afe13e580271f9803b31))
* **web:** Advanced packs — distributions & nullability (P2b-1) ([93c50c9](https://github.com/dbsprout/dbsprout/commit/93c50c9781467f712dd24e6cd2fe203358cbeecd))
* **web:** Advanced packs — distributions & nullability in the Inspector (P2b-1) ([e699308](https://github.com/dbsprout/dbsprout/commit/e699308b4cf84651f55c268fe41422618be202aa))
* **web:** AI spec-assist — cloud key-entry UX (P4-11) ([b59271a](https://github.com/dbsprout/dbsprout/commit/b59271a890b84a741297ec9d4adfc68db9c1c840))
* **web:** AI spec-assist — cloud key-entry UX (P4-11) ([73cb890](https://github.com/dbsprout/dbsprout/commit/73cb8907e89678e8fdfb6d9ffb14c96f6524c98b))
* **web:** Configure — activate dtype filter wiring (P4-12) ([f4d686f](https://github.com/dbsprout/dbsprout/commit/f4d686f56871cdd2115018a1c2919e7b01474023))
* **web:** Configure — activate dtype filter wiring (P4-12) ([1a1ac88](https://github.com/dbsprout/dbsprout/commit/1a1ac88011fdd42cca2ce353e07389e0d438d9df))
* **web:** Configure — column grid virtualization (P4-2) ([abe000e](https://github.com/dbsprout/dbsprout/commit/abe000ee268cd142988b2311538c12aaee7d501c))
* **web:** Configure — dtype-filtered generator dropdowns (P4-1) ([0267689](https://github.com/dbsprout/dbsprout/commit/0267689757a5c04116376b3b039888e0fcce7b07))
* **web:** Configure — dtype-filtered generator dropdowns (P4-1) ([9130fc4](https://github.com/dbsprout/dbsprout/commit/9130fc4de56eea34aef2db19bc66c1ed343fbda0))
* **web:** Configure — Inspector/grid dual-edit refresh (P4-3) ([7a2a154](https://github.com/dbsprout/dbsprout/commit/7a2a154ab50845e9298598546bba0fed32f14cd8))
* **web:** Configure — Inspector/grid dual-edit refresh (P4-3) ([d6014c4](https://github.com/dbsprout/dbsprout/commit/d6014c4a33f1cce0d35e28bc8758b8230d1224bd))
* **web:** Configure — per-column generator grid + Inspector + live preview (P1b-2) ([3ce5bf8](https://github.com/dbsprout/dbsprout/commit/3ce5bf872df8412c91ac2fdcbabd613f6acecab3))
* **web:** Configure — spec API client + per-table row counts (P1b-1) ([7727e53](https://github.com/dbsprout/dbsprout/commit/7727e53548f2c0c7f95013ccf5a39bd4922c0750))
* **web:** Configure panel with per-table row counts (P1b-1) ([f1bae7b](https://github.com/dbsprout/dbsprout/commit/f1bae7b062b7f2c16e5fde12c78ab366a4d5317f))
* **web:** Connect — URL-paste field auto-fill (P4-8) ([c66da32](https://github.com/dbsprout/dbsprout/commit/c66da32c994c7e5ccddc9e4b990760b7f5b56f90))
* **web:** connection-url builder + multipart upload client (P1a-2b) ([5338e2b](https://github.com/dbsprout/dbsprout/commit/5338e2bf3e0ed7533565fc167c87566c746e850a))
* **web:** Connectivity — saved connections (.dbsprout/connections.toml) (P2a-2) ([3162478](https://github.com/dbsprout/dbsprout/commit/316247845f411026760bdd72aeaeea63bebda852))
* **web:** Connectivity — saved connections (.dbsprout/connections.toml) (P2a-2) ([9c15947](https://github.com/dbsprout/dbsprout/commit/9c15947cf5df102f65aaa43bedc3143f00098217))
* **web:** Connectivity — SSH tunnel ([ssh] extra) (P2a-3) ([a30649a](https://github.com/dbsprout/dbsprout/commit/a30649a9c737c1f1a4a166a824d165bcc3602351))
* **web:** Connectivity — SSH tunnel ([ssh] extra) (P2a-3) ([df6d9b6](https://github.com/dbsprout/dbsprout/commit/df6d9b6a8460ab90f859472e280c6dff86ec00ce))
* **web:** Connectivity — SSL/schema/params connect fields (P2a-1) ([f7b66b8](https://github.com/dbsprout/dbsprout/commit/f7b66b886929dc2f6cb7ab971e7b90740aac5bc9))
* **web:** Connectivity — SSL/schema/params connect fields (P2a-1) ([fa83e91](https://github.com/dbsprout/dbsprout/commit/fa83e91df9d41919a59920af78a862cf05cc2b98))
* **web:** Cross-panel selection store — Validate drill-to-cell (P4-7) ([f2f5141](https://github.com/dbsprout/dbsprout/commit/f2f514121734af4a7f479df685c6908d84a8c37d))
* **web:** Cross-panel selection store — Validate drill-to-cell (P4-7) ([03c4bad](https://github.com/dbsprout/dbsprout/commit/03c4badf7a750bd341031935e4ab8f0f3a529d4a))
* **web:** cutover — remove legacy HTMX UI, JSON-only API, / → /app (P1c-5) ([9d2266d](https://github.com/dbsprout/dbsprout/commit/9d2266d0ee589d8b82ca10f6e80f49e27e15d252))
* **web:** cutover — remove legacy HTMX UI, JSON-only API, / → /app (P1c-5) ([67b9e43](https://github.com/dbsprout/dbsprout/commit/67b9e43f334c8067eb83ba75b0e1d1a7c3182ca1))
* **web:** export StepId + STEP_IDS for the focus overlay (P3-2) ([1fc6e32](https://github.com/dbsprout/dbsprout/commit/1fc6e3274321d2a646bc83b304a219191532487b))
* **web:** first-run Sample offer reusing loadSample (P3-2) ([ac0f97b](https://github.com/dbsprout/dbsprout/commit/ac0f97bb77137c4a31a6a74f03f49ea258e3bb19))
* **web:** Generate — cancel affordance (P4-6) ([0c8a25e](https://github.com/dbsprout/dbsprout/commit/0c8a25e08e0a9db4c99403b8c9cf6682433d7026))
* **web:** Generate — cancel affordance (P4-6) ([8cd04f5](https://github.com/dbsprout/dbsprout/commit/8cd04f57beb2e29979e6b8e2b81804b26e36b182))
* **web:** Generate — engine select + run + progress console + result preview (P1b-3) ([4ba1275](https://github.com/dbsprout/dbsprout/commit/4ba1275565b58614516ae37074175c1a8995903a))
* **web:** Generate — engine select + run + progress console + result preview (P1b-3) ([5421e8e](https://github.com/dbsprout/dbsprout/commit/5421e8edb8cbfe9e1c399a6daaf21883715bd3ac))
* **web:** Generate — richer job-result envelope (P4-4) ([f34019c](https://github.com/dbsprout/dbsprout/commit/f34019cce6ae46158fe8093f1b6d28b46c7b426d))
* **web:** Generate — richer job-result envelope (P4-4) ([1943c5c](https://github.com/dbsprout/dbsprout/commit/1943c5c0ea2015f545f97b2e98937c421c32c7f4))
* **web:** Generate — WebSocket progress streaming (P4-5) ([77a6308](https://github.com/dbsprout/dbsprout/commit/77a63083bdc732fa19e6d8d2f0e3b268a71ae38d))
* **web:** Generate — WebSocket progress streaming (P4-5) ([5bb6c46](https://github.com/dbsprout/dbsprout/commit/5bb6c46bdce5278b48ada3a5943b50dcdd00ebcc))
* **web:** GET /api/samples + POST /api/schema/sample (P1a) ([d875796](https://github.com/dbsprout/dbsprout/commit/d875796e1734fd94b6e80caf37e8f44e0d4a9f26))
* **web:** Guided focus overlay + coach copy + first-run sample (P3-2) ([ba0abe2](https://github.com/dbsprout/dbsprout/commit/ba0abe248d3b7f88bd65bc0258d76c503498f35d))
* **web:** Guided mode engine — toggle + stepper + gating (P3-1) ([f3069f4](https://github.com/dbsprout/dbsprout/commit/f3069f40d6d5f55808bd9b7198e9acd04670aaa0))
* **web:** guided-only stepper with reactive gated Next/Back (P3-1) ([6f5a1e7](https://github.com/dbsprout/dbsprout/commit/6f5a1e7eb5484e2992fe75dbe57e4ed8d49ed0fb))
* **web:** GuidedSection focus-overlay wrapper (P3-2) ([43814c5](https://github.com/dbsprout/dbsprout/commit/43814c54e59aae03a06082c74c6fd02a72c1a986))
* **web:** mode context with localStorage persistence (P3-1) ([a5a9f77](https://github.com/dbsprout/dbsprout/commit/a5a9f77b5e8e2ad3e47495848bfdf22c1b70e8d7))
* **web:** mode toggle + stepper wired into AppShell header (P3-1) ([a853fc4](https://github.com/dbsprout/dbsprout/commit/a853fc464e7d8705c7a3ae58fb25f205d76ca6e9))
* **web:** mount ConfigurePanel in App Configure section (P1b-2) ([cdd5466](https://github.com/dbsprout/dbsprout/commit/cdd5466bf6ba0636750beb3605b2126b45d1d22b))
* **web:** Output — Export panel (SQL/CSV/JSON/Parquet) (P1c-1) ([3e6da59](https://github.com/dbsprout/dbsprout/commit/3e6da592097d2429c2860755a4c0019442c85965))
* **web:** Output — Export panel (SQL/CSV/JSON/Parquet) (P1c-1) ([cdfcc06](https://github.com/dbsprout/dbsprout/commit/cdfcc06bcb3a796fc3595b3cb801e9ecdbe7d503))
* **web:** Output — Insert panel with write-guard confirm (P1c-2) ([0a4ba34](https://github.com/dbsprout/dbsprout/commit/0a4ba3450f63ccef41d60c7973708924670158fa))
* **web:** Output — Insert panel with write-guard confirm (P1c-2) ([7d18950](https://github.com/dbsprout/dbsprout/commit/7d189509b68a74e5ffaefe2f4a51eb3f276b60c1))
* **web:** P1a connect/schema backend (test-connection, paste, samples) ([c7568a2](https://github.com/dbsprout/dbsprout/commit/c7568a2ac1ab670c140c5f31bfb9d9a7101085ee))
* **web:** per-step coach copy + CoachPanel (P3-2) ([73a529a](https://github.com/dbsprout/dbsprout/commit/73a529aed856e0465858788e54fdc44d8352963c))
* **web:** POST /api/connect/test probe (P1a) ([ba90cdb](https://github.com/dbsprout/dbsprout/commit/ba90cdbd950bcb40a90833e55da12d314eaab544))
* **web:** POST /api/schema/paste (P1a) ([9f97b93](https://github.com/dbsprout/dbsprout/commit/9f97b93de510d2c5ed3ac1cc51d404751806b437))
* **web:** Runs/Quality/Costs panels + JSON state endpoints (P1c-4) ([86e03b7](https://github.com/dbsprout/dbsprout/commit/86e03b78129e528bec7cf02f934a87b2d9e07779))
* **web:** Runs/Quality/Costs panels + JSON state endpoints (P1c-4) ([b8b316f](https://github.com/dbsprout/dbsprout/commit/b8b316f7c53568e9829094d7e90826b5c929c9fd))
* **web:** scaffold React+Vite+TS Workbench frontend ([adbe275](https://github.com/dbsprout/dbsprout/commit/adbe2756c1b634185522e83d1ee849cb8a16cfcb))
* **web:** serve Workbench SPA at /app with placeholder fallback ([b6d5e40](https://github.com/dbsprout/dbsprout/commit/b6d5e40c149c30d0e59d08c92de46eef0f8e6280))
* **web:** spec + generators API client (P1b-1) ([e9e60ac](https://github.com/dbsprout/dbsprout/commit/e9e60ace639d7bcc39869467eaf6e63387db1554))
* **web:** SSH — bastion-failure friendly typed error (P4-9) ([094ad08](https://github.com/dbsprout/dbsprout/commit/094ad08f1debbe8c7a730a2b73b6c8e1f912ae48))
* **web:** SSH — bastion-failure friendly typed error (P4-9) ([c75a5c5](https://github.com/dbsprout/dbsprout/commit/c75a5c5fa0ff714bdcacc437fdc1df899fd3cfbd))
* **web:** SSH — incomplete-block inline validation (P4-10) ([bcecd11](https://github.com/dbsprout/dbsprout/commit/bcecd1118925ec1c9ae759884b4272130186a47d))
* **web:** SSH — incomplete-block inline validation (P4-10) ([5479508](https://github.com/dbsprout/dbsprout/commit/5479508ac0ac6d820a3641dc0ad07d02f474e084))
* **web:** tabbed Start panel + connect/upload/paste forms (P1a-2b) ([92a1bee](https://github.com/dbsprout/dbsprout/commit/92a1bee1349423b5f43244e46841775bdfe33b09))
* **web:** trigger first-run sample on advanced-&gt;guided toggle, not just initial mount (P3-2) ([4ce220b](https://github.com/dbsprout/dbsprout/commit/4ce220b14149262738e59cdd83a27cfc2497d327))
* **web:** typed API client + TanStack Query provider (P1a-2a) ([a462ac2](https://github.com/dbsprout/dbsprout/commit/a462ac224d3fc90a52186c01eb470a411e12d78d))
* **web:** Validate panel — integrity + fidelity + detection (P1c-3) ([751bdd4](https://github.com/dbsprout/dbsprout/commit/751bdd4fd475f12fc4226fd589f518fd1c5b625c))
* **web:** Validate panel — integrity + fidelity + detection (P1c-3) ([d2c22c5](https://github.com/dbsprout/dbsprout/commit/d2c22c572aac338f591a662cefabae2e174d5d10))
* **web:** virtualize ColumnGrid with react-virtual (P4-2) ([4e67c68](https://github.com/dbsprout/dbsprout/commit/4e67c68457afd9ed8d787dddadfea67a40b84c74))
* **web:** wire focus overlay + coach + first-run into App/AppShell (P3-2) ([16d15cb](https://github.com/dbsprout/dbsprout/commit/16d15cbc6c0c6d8a46ecac87c7928e2a6faac11f))
* **web:** Workbench frontend foundation + sample→schema (P1a-2a) ([4b1443e](https://github.com/dbsprout/dbsprout/commit/4b1443ef0e38364317d5583421c57d0d2bb2debe))
* **web:** Workbench shell + sample picker + schema tree (P1a-2a) ([a827f65](https://github.com/dbsprout/dbsprout/commit/a827f65305e645b50df50aa83b53c267782736ef))
* **web:** Workbench SPA — P0 foundation (DBS-181) ([c468733](https://github.com/dbsprout/dbsprout/commit/c468733582a3ec8c799ab3a99b1c2cf8300d0d60))
* **web:** Workbench Start screen — connect form + upload + paste (P1a-2b) ([5542e83](https://github.com/dbsprout/dbsprout/commit/5542e83b2c4aa2313ca121aaebc0da81caa4473a))
* **web:** wrap App in ModeProvider + advanced-mode regression guard (P3-1) ([f6c9007](https://github.com/dbsprout/dbsprout/commit/f6c9007d9baabd1369fc241415a20ce543ec30c6))


### Bug Fixes

* **S-148:** make TUI progress on_mount idempotent to stop row duplication ([78ff20c](https://github.com/dbsprout/dbsprout/commit/78ff20c31c35cde43dfd58872f1a3bfe23be9210))
* **S-148:** make TUI progress on_mount idempotent to stop row duplication ([13f76ac](https://github.com/dbsprout/dbsprout/commit/13f76acce9ce854f89b6be12b206975c08107d38))
* **S-149:** make cli_diff --file path assertions width-independent ([6ffebac](https://github.com/dbsprout/dbsprout/commit/6ffebac81b827e98e3fd077d9695f575b3c2d4ce))
* **S-149:** make cli_diff --file path assertions width-independent ([e4ac789](https://github.com/dbsprout/dbsprout/commit/e4ac789921d935a2907eed4dd5accc413d8ebca3))
* **S-153:** consolidate train extractor to a single DB engine ([c79e692](https://github.com/dbsprout/dbsprout/commit/c79e692d93dc8425e14452a71bb525939b23cc1d))
* **S-153:** consolidate train extractor to a single DB engine ([1e33f08](https://github.com/dbsprout/dbsprout/commit/1e33f0842b3232445f97f18337387a09194851df))
* **S-153:** repoint extractor helper-test patches to introspect_engine ([487c2d1](https://github.com/dbsprout/dbsprout/commit/487c2d18e760a9bc1b4f1ff71be989efacaa0169))
* **web:** consistent sample dialect + accurate saas description (P1a) ([0d2d374](https://github.com/dbsprout/dbsprout/commit/0d2d374f4621047f2a4c63a81ef0c789d7242541))
* **web:** spec-assist never-500 — catch broad provider failures (P2b-3) ([ea22fd3](https://github.com/dbsprout/dbsprout/commit/ea22fd3fa92f6c75ed6d187769f6a1b404ca3bc6))
* **web:** stepper re-renders on in-place query updates via monotonic version (P3-1) ([eb1aeb7](https://github.com/dbsprout/dbsprout/commit/eb1aeb79d31a84267e670183b386d1f90f0c0950))


### Code Refactoring

* **S-150:** consolidate duplicated _detect_direct_dialect ([f8a85b6](https://github.com/dbsprout/dbsprout/commit/f8a85b6f3dbed83adae03077626e25332f530cd0))
* **S-150:** consolidate duplicated _detect_direct_dialect ([42c9c7a](https://github.com/dbsprout/dbsprout/commit/42c9c7a97c3f391d525873fc691cdc615fb0dd19))
* **web:** drop unused react-query dep, pin node&gt;=20 ([2e2963c](https://github.com/dbsprout/dbsprout/commit/2e2963c4d810bc4f9880130321022f091cc26ac0))


### Miscellaneous

* release 0.2.0 ([d20952e](https://github.com/dbsprout/dbsprout/commit/d20952e0ad30716999b40794f3f21cac7cd58f2d))
* relicense from MIT to AGPL-3.0-or-later ([a9a15a6](https://github.com/dbsprout/dbsprout/commit/a9a15a6bd4b2c7f678122875b5ad18cc1f7594f9))
* relicense from MIT to AGPL-3.0-or-later ([75f0bf4](https://github.com/dbsprout/dbsprout/commit/75f0bf40322a3a3b2180851549ebc19f8eefd318))
* **web:** add @tanstack/react-virtual dep (P4-2) ([9df4ecf](https://github.com/dbsprout/dbsprout/commit/9df4ecfef15d6ad9cf3aae003d380458434f2be2))


### CI/CD

* **S-151:** add web-router collection-check gate ([717b382](https://github.com/dbsprout/dbsprout/commit/717b3827bf72cfe4d11377faa9c4c7c438df5014))
* **S-151:** add web-router collection-check gate ([262b3c6](https://github.com/dbsprout/dbsprout/commit/262b3c6a25e6afdf9625a034b5ebfdc44cc780d1))
* **web:** frontend build/test job + bundle SPA in release wheel ([8ea6ec3](https://github.com/dbsprout/dbsprout/commit/8ea6ec3c4147755f72faf6502716b24868f25dbd))


### Tests

* **web:** add vitest coverage tooling (P1b-2) ([b167baf](https://github.com/dbsprout/dbsprout/commit/b167bafb21cae7f9e27fbe7d97453057dbd6ce6d))
* **web:** cover localStorage failure paths in ModeProvider (P3-1) ([ce9b907](https://github.com/dbsprout/dbsprout/commit/ce9b907d09def047ae40130e1c6045a18d12160a))
* **web:** cover Upload/Paste forms + exclude type-only module for &gt;=80% coverage (P1b-2) ([f3e53b1](https://github.com/dbsprout/dbsprout/commit/f3e53b1bc92a16d186f223c7c1e1b65b7c8d0598))
* **web:** fix tsc tuple-index error in ConfigurePanelAdvanced mock ([d3d91d3](https://github.com/dbsprout/dbsprout/commit/d3d91d36bf6048a9363adf4ac3a8f451a095650f))
* **web:** set guided step via effect to avoid render-phase setState (P3-2) ([7feef69](https://github.com/dbsprout/dbsprout/commit/7feef69309585ea777e0cb1e98670076a2fe2fff))
* **web:** type PasteForm fetch mock for strict tsc (P1b-2) ([db506a6](https://github.com/dbsprout/dbsprout/commit/db506a6f7d29535c6c14db601e61dbc3d240943f))
* **web:** Vitest smoke test for Workbench App ([ed6c987](https://github.com/dbsprout/dbsprout/commit/ed6c9875958ab58bbba959fb977c8b63cbb93001))

## [0.1.8](https://github.com/dbsprout/dbsprout/compare/v0.1.7...v0.1.8) (2026-05-28)


### Features

* **DBS-112:** plugin runtime hardening [S-095] ([09c2324](https://github.com/dbsprout/dbsprout/commit/09c2324cb434fe7a36a49823c747580637969410))
* **DBS-112:** plugin runtime hardening [S-095] ([9141368](https://github.com/dbsprout/dbsprout/commit/9141368818e92d59e2d443b0d800f862d8ad9460))
* **DBS-116:** benchmark regression gate [S-096] ([d1cf4bc](https://github.com/dbsprout/dbsprout/commit/d1cf4bc49bc4d2fa55077ba5b397b5d28b94b188))
* **DBS-116:** benchmark regression gate [S-096] ([b87b0f7](https://github.com/dbsprout/dbsprout/commit/b87b0f7256ec8fbc5851e7880726ba893f66fe87))
* **DBS-90:** TUI app skeleton [S-086] ([4c89ee8](https://github.com/dbsprout/dbsprout/commit/4c89ee8a3d43100191643ec72dbecfa47ec43ad4))
* **DBS-90:** TUI app skeleton [S-086] ([4287850](https://github.com/dbsprout/dbsprout/commit/4287850bebe6360d4bbe7e36e52b00e20078b4fc))
* **DBS-91:** TUI progress screen [S-087] ([f1eff16](https://github.com/dbsprout/dbsprout/commit/f1eff165eff52e93eb04e98bb052098a71d25afc))
* **DBS-91:** TUI progress screen [S-087] ([7f50694](https://github.com/dbsprout/dbsprout/commit/7f5069464d999c06e6ee3ed10fd044d5a8f49d68))
* **DBS-92:** TUI schema browser [S-088] ([f798eb0](https://github.com/dbsprout/dbsprout/commit/f798eb0ece03e682fd1b09d9dd2d0855fd10e58a))
* **DBS-92:** TUI schema browser [S-088] ([a8d5711](https://github.com/dbsprout/dbsprout/commit/a8d57110b3135af442430b82c00811530b043b15))
* **DBS-93:** TUI quality display [S-089] ([33c642c](https://github.com/dbsprout/dbsprout/commit/33c642c4903a8ad2d509ac3dddc71084dbd62d6d))
* **DBS-93:** TUI quality display [S-089] ([0df0105](https://github.com/dbsprout/dbsprout/commit/0df0105406d49f005d15a093148e0b024ec28cbe))
* **DBS-94:** FastAPI web dashboard skeleton [S-090] ([5afe30c](https://github.com/dbsprout/dbsprout/commit/5afe30cfc87b8c729fc941840fb201f6c5c33cb8))
* **DBS-94:** FastAPI web dashboard skeleton [S-090] ([c29db57](https://github.com/dbsprout/dbsprout/commit/c29db571275cc08e2e85b74d4bfd0189d03bf6d1))
* **DBS-95:** web schema ERD view [S-091] ([b0be38d](https://github.com/dbsprout/dbsprout/commit/b0be38dad9f85e2ebbdf18bd0f4955a8d47c9ecb))
* **DBS-95:** web schema ERD view [S-091] ([a7ced0c](https://github.com/dbsprout/dbsprout/commit/a7ced0cb0f99e1481cade10890c451f8e3e62f8d))
* **DBS-96:** web SSE progress view [S-092] ([25addf5](https://github.com/dbsprout/dbsprout/commit/25addf558309b7c45bf5732aaf1270749f5bc780))
* **DBS-96:** web SSE progress view [S-092] ([0d08106](https://github.com/dbsprout/dbsprout/commit/0d0810667678370e76662759a55e6389d8fbb8dd))
* **DBS-97:** web quality/preview/cost/history views [S-093] ([7bec0c0](https://github.com/dbsprout/dbsprout/commit/7bec0c0ea695ba1cbac04225cf22ad2fb21335f3))
* **DBS-97:** web quality/preview/cost/history views [S-093] ([a93592e](https://github.com/dbsprout/dbsprout/commit/a93592ed5eb374ab5115a8b87ee94e00424c77d4))
* **generate:** add nonce to column_seed for per-column re-roll (S-130) ([a6e4016](https://github.com/dbsprout/dbsprout/commit/a6e401632782b61c84f6a9dc76311cb05535347b))
* **generate:** add progress_callback + cancel_token to orchestrate (S-107) ([8e9c80a](https://github.com/dbsprout/dbsprout/commit/8e9c80a61d53bcb69b8fefe7a47efdaf379e93f1))
* **generate:** add progress_callback + cancel_token to orchestrate (S-107) ([ed72968](https://github.com/dbsprout/dbsprout/commit/ed7296885646dca7feb6f8fa216292c1725c2422))
* **generate:** add regenerate_column with PK/FK-ref restrictions (S-129) ([6019434](https://github.com/dbsprout/dbsprout/commit/60194341449d4bcd4b5741846867529b49f51a67))
* **generate:** nonce-based column_seed for per-column re-roll (S-130) ([c52982a](https://github.com/dbsprout/dbsprout/commit/c52982a3f1d23d2248ae2c9d4ba4ebe3822b4b09))
* **generate:** regenerate_column with PK/FK-ref restrictions (S-129) ([d2c4527](https://github.com/dbsprout/dbsprout/commit/d2c452744e771bc6f8cfd66d88daddd1cd69643b))
* **generate:** regenerate_table() — PK-stable single-table re-roll (S-128) ([e05f42e](https://github.com/dbsprout/dbsprout/commit/e05f42e5f07952e82f3d63b8135f47ce86c7bc76))
* **generate:** regenerate_table() — PK-stable single-table re-roll (S-128) ([8d0dbaa](https://github.com/dbsprout/dbsprout/commit/8d0dbaaee4c378502bbce9ab6739ecbce00225b2))
* **migrate:** cap total migration files in parsers [S-105] ([5039935](https://github.com/dbsprout/dbsprout/commit/5039935d43f85d0af8ba7afad6fddeb62755693d))
* **migrate:** cap total migration files in parsers [S-105] ([acd21d3](https://github.com/dbsprout/dbsprout/commit/acd21d337d15847fe9e007ff3d4e7e58a52165c6))
* **output:** column_update writer (UPDATE…WHERE pk) (S-138) ([9db01d5](https://github.com/dbsprout/dbsprout/commit/9db01d58a416bad6a72b28b2cf54589a10559ff2))
* **output:** column_update writer (UPDATE…WHERE pk) (S-138) ([3d35d25](https://github.com/dbsprout/dbsprout/commit/3d35d25b6ffa4442f48891e66d870eca4e6ee254))
* **web:** add fidelity + detection blocks to validate report (S-134) ([c24169d](https://github.com/dbsprout/dbsprout/commit/c24169d0f0c5852d4929aa62e6dd6469a6866d44))
* **web:** add in-memory Workspace session object (S-111) ([a5e594a](https://github.com/dbsprout/dbsprout/commit/a5e594ae14a6a1fdc76338a02979a02cc4d0353d))
* **web:** add in-memory Workspace session on app.state (S-111) ([cc0c2db](https://github.com/dbsprout/dbsprout/commit/cc0c2dbbbabae753fcaab0b24e2442469db2b338))
* **web:** add in-process JobManager with cancel (S-108) ([a9c6cdb](https://github.com/dbsprout/dbsprout/commit/a9c6cdbff51594109996b7b447819c5362ddb433))
* **web:** add in-process JobManager with cancel (S-108) ([5c86013](https://github.com/dbsprout/dbsprout/commit/5c86013154a489fe65c22346d24f746908554a13))
* **web:** add Open-in-Studio handoff on wizard step wrapper (S-144) ([f473ceb](https://github.com/dbsprout/dbsprout/commit/f473ceb2a512e92776d5528092ef1f87f1e405d2))
* **web:** add Open-in-Studio handoff on wizard step wrapper (S-144) ([c63a38b](https://github.com/dbsprout/dbsprout/commit/c63a38b90565df0543533ceb6fdd1f9e3a0e25b4))
* **web:** add POST /api/regenerate sync/job route (S-131) ([f522fc1](https://github.com/dbsprout/dbsprout/commit/f522fc143889df0de140ac83892c17eb1c391617))
* **web:** add ProgressHub + /ws/jobs WebSocket route (S-109) ([354abe9](https://github.com/dbsprout/dbsprout/commit/354abe9e440839fd97e23caf229e70f9eb92f2ea))
* **web:** add S-140 export error envelopes ([8650b7d](https://github.com/dbsprout/dbsprout/commit/8650b7db567ae695f85948ccc252be74091341f8))
* **web:** add Workspace.peek_target_url for internal credential scrubbing (S-124) ([debd336](https://github.com/dbsprout/dbsprout/commit/debd336612598f6f091576323201182b59d383de))
* **web:** auto heuristic spec on wizard Step 3 + opt-in LLM (S-145) ([a570d4f](https://github.com/dbsprout/dbsprout/commit/a570d4f2dae03d1606b9eef97d4f59c48f235d14))
* **web:** auto heuristic spec on wizard Step 3 + opt-in LLM path (S-145) ([285e39d](https://github.com/dbsprout/dbsprout/commit/285e39db6cfb694722815e55798cfcce554e5cba))
* **web:** cancel running generate job (S-126) ([1485ac3](https://github.com/dbsprout/dbsprout/commit/1485ac3f26c63b49a52e9844c49eb012fd9f5f92))
* **web:** cancel running generate job via POST /api/jobs/{id}/cancel (S-126) ([52e68e4](https://github.com/dbsprout/dbsprout/commit/52e68e4afe9a0ac8a192ec4652506874733112a2))
* **web:** consolidate tooltip/help system + generator examples (S-146) ([b8af308](https://github.com/dbsprout/dbsprout/commit/b8af308d20f5c0a522c450857da6b0a515887754))
* **web:** consolidate tooltip/help system + generator examples (S-146) ([bc3be29](https://github.com/dbsprout/dbsprout/commit/bc3be29224d3a0b8c68526d50acdad007afdc06e))
* **web:** explicit insert method select (auto/batch/copy) (S-141) ([8222520](https://github.com/dbsprout/dbsprout/commit/8222520c40919f87292281ebd7cec396dab60db9))
* **web:** explicit insert method select (auto/batch/copy) (S-141) ([ab16e20](https://github.com/dbsprout/dbsprout/commit/ab16e20b2abffccb8571aaa7dc1093f6afd340b7))
* **web:** fidelity + detection blocks in validate report (S-134) ([e585a1b](https://github.com/dbsprout/dbsprout/commit/e585a1baaa47e669b83545db73d91d0525931825))
* **web:** friendly typed-envelope errors for /api/connect + /api/schema/load (S-116) ([b2bbd5a](https://github.com/dbsprout/dbsprout/commit/b2bbd5ad8eb0b0de1f2b5e703f6d8e538294f62e))
* **web:** friendly typed-envelope errors for /api/connect + /api/schema/load [S-116] ([42da11e](https://github.com/dbsprout/dbsprout/commit/42da11eaa26aa9489a610d86ceb1374edaff2ee1))
* **web:** GET /api/preview/{table} bounded sample (S-147) ([64ab8b3](https://github.com/dbsprout/dbsprout/commit/64ab8b39be456ef719ada54dc9d47c8f6a518fac))
* **web:** GET /api/preview/{table} bounded sample (S-147) ([dff0a8e](https://github.com/dbsprout/dbsprout/commit/dff0a8e3f325f17f79ef3c70df13dc65867d2ad3))
* **web:** GET /api/schema + ERD review (S-115) ([c178f58](https://github.com/dbsprout/dbsprout/commit/c178f5839c0285ebb91d6825159984aede90dda6))
* **web:** GET /api/schema + ERD review (S-115) ([fbc8e25](https://github.com/dbsprout/dbsprout/commit/fbc8e25d7c27287f74bcdcdccb6dbdc9bf68c966))
* **web:** GET /api/spec + spec grid with interactive column headers (S-118) ([b439332](https://github.com/dbsprout/dbsprout/commit/b439332a0e81d4682f21cceebad4528b5b4c12c3))
* **web:** GET /api/spec + spec grid with interactive column headers (S-118) ([9193d99](https://github.com/dbsprout/dbsprout/commit/9193d9962dd91e9ee3e8ef7452efde028c865ff7))
* **web:** inline tooltips on config fields + generator descriptions (S-123) ([d541ef5](https://github.com/dbsprout/dbsprout/commit/d541ef520da2acf8f8a69beed071f198b88fca70))
* **web:** inline tooltips on config fields + generator descriptions (S-123) ([0795301](https://github.com/dbsprout/dbsprout/commit/0795301ba6e18a64684ffc6b641510458ded174f))
* **web:** interactive ERD zoom/pan/click-detail and large-schema controls [S-091-F1] ([5aff449](https://github.com/dbsprout/dbsprout/commit/5aff4499aa4a1067677c9c50f5e60cd2c1edf5e1))
* **web:** interactive ERD zoom/pan/click-detail and large-schema controls [S-091-F1] ([b07fb52](https://github.com/dbsprout/dbsprout/commit/b07fb526d6db309525e5fb56e16f0cb1829d5732))
* **web:** JobManager publishes progress to the hub (S-109) ([cdafb0d](https://github.com/dbsprout/dbsprout/commit/cdafb0d231cb4f594f5c44ea287705dfbb44a532))
* **web:** live progress console with per-table bars + rows/sec (S-125) ([c377542](https://github.com/dbsprout/dbsprout/commit/c3775428d7fb9aa987d8693b4faf7d8e947b98af))
* **web:** live progress console with per-table bars + rows/sec (S-125) ([ef34cca](https://github.com/dbsprout/dbsprout/commit/ef34cca1b5f95e5788b0614ebbc87097bbc0973a))
* **web:** method-picker + GET /api/generators (S-120) ([fb1022d](https://github.com/dbsprout/dbsprout/commit/fb1022da1d91d359d229501a20f5b4dc93a7f9e2))
* **web:** method-picker UI + GET /api/generators (S-120) ([8c1fd24](https://github.com/dbsprout/dbsprout/commit/8c1fd2490963ec1153b1340fda1017c04729522f))
* **web:** parser auto-detect for /api/schema/load (S-114) ([9845318](https://github.com/dbsprout/dbsprout/commit/984531846af1124a382fb3ebcdddb1b7c95c0c5e))
* **web:** parser auto-detect for /api/schema/load (S-114) ([4b98278](https://github.com/dbsprout/dbsprout/commit/4b982784082c0250c9de6bd59ff9335d59fe87a0))
* **web:** per-column method swap -&gt; instant re-preview (S-132) ([29e5ca3](https://github.com/dbsprout/dbsprout/commit/29e5ca3b0e35cc98f4490344ec3784b498c96616))
* **web:** per-column method swap -&gt; instant re-preview (S-132) ([e17d8ad](https://github.com/dbsprout/dbsprout/commit/e17d8ad4722a9dee5074f582f21da766fd9cde8d))
* **web:** persist completed jobs via StateDB; job history (S-110) ([cef1802](https://github.com/dbsprout/dbsprout/commit/cef18020c054073e65868b339a65775e4d9dee58))
* **web:** persist completed jobs via StateDB; job history (S-110) ([dcdcd10](https://github.com/dbsprout/dbsprout/commit/dcdcd100494719175552e9a7582a71c5e0258396))
* **web:** persist spec edits to cache keyed by schema_hash (S-122) ([f7550a4](https://github.com/dbsprout/dbsprout/commit/f7550a43bcd2c84e3ad3e26e58fdbbed486e9641))
* **web:** persist spec edits to cache keyed by schema_hash (S-122) ([4f44905](https://github.com/dbsprout/dbsprout/commit/4f44905082d9d0d95193ff68ed3d7872728f7da0))
* **web:** POST /api/connect schema introspection (S-112) ([5d491d3](https://github.com/dbsprout/dbsprout/commit/5d491d3fbc570d2ae8c76900b79bad9ba0971bc1))
* **web:** POST /api/connect schema introspection (S-112) ([306199c](https://github.com/dbsprout/dbsprout/commit/306199cb5a3d04dd4e7f060c8e07acd71184aebe))
* **web:** POST /api/export SQL/CSV/JSON/Parquet via writers (S-140) ([27378eb](https://github.com/dbsprout/dbsprout/commit/27378eb4967127b14c18212c0acadab1352c450b))
* **web:** POST /api/export SQL/CSV/JSON/Parquet via writers (S-140) ([3c4cfe0](https://github.com/dbsprout/dbsprout/commit/3c4cfe0dad1b06641a200b306ded1f8a125dc1a9))
* **web:** POST /api/generate background job (S-124) ([298d3a1](https://github.com/dbsprout/dbsprout/commit/298d3a1f26bb829effc2cf8a54247b0fe8d3de9a))
* **web:** POST /api/generate submits a background job (S-124) ([37e1821](https://github.com/dbsprout/dbsprout/commit/37e18214f085cd2cc2a0c21592164b1478ec3cb7))
* **web:** POST /api/insert all + by-table via writers (S-136) ([4eb0d30](https://github.com/dbsprout/dbsprout/commit/4eb0d30dce08a4ff74c94421ea291fe016418747))
* **web:** POST /api/insert all + by-table via writers as a job (S-136) ([8d3e253](https://github.com/dbsprout/dbsprout/commit/8d3e253b2124b624e59c33665451783a62dc9068))
* **web:** POST /api/regenerate sync/job route (S-131) ([614f943](https://github.com/dbsprout/dbsprout/commit/614f94366804ba2fa40b36ec0c5318a8b10365eb))
* **web:** POST /api/schema/load file upload + parsers (S-113) ([5be5937](https://github.com/dbsprout/dbsprout/commit/5be5937c7b83bb6f864fc45bfe388e100358d097))
* **web:** POST /api/schema/load file upload + parsers (S-113) ([2361c3e](https://github.com/dbsprout/dbsprout/commit/2361c3e581df9a7bda187c9f03392c9e89c726ae))
* **web:** POST /api/update-column endpoint wiring (S-139) ([fa7bf5d](https://github.com/dbsprout/dbsprout/commit/fa7bf5d45aa6aa682121fbf06c401134ea247a1e))
* **web:** POST /api/update-column endpoint wiring (S-139) ([4a3b3e7](https://github.com/dbsprout/dbsprout/commit/4a3b3e7371e5a483cfd5e43a24b48630a0df58d5))
* **web:** POST /api/validate integrity report + HTMX panel (S-133) ([52a3d6e](https://github.com/dbsprout/dbsprout/commit/52a3d6e35d287cf2314872de93661d0751e75899))
* **web:** POST /api/validate integrity report + HTMX panel (S-133) ([1990ee8](https://github.com/dbsprout/dbsprout/commit/1990ee83b259064a6cf85a7c38dc1fbabb8a9563))
* **web:** progress hub + /ws/jobs WebSocket (S-109) ([ae87e7d](https://github.com/dbsprout/dbsprout/commit/ae87e7d0e115814b08fa6ab1447a4777cc5e42b4))
* **web:** PUT /api/spec/tables/{t} — row_count edit (S-121) ([cd71d3f](https://github.com/dbsprout/dbsprout/commit/cd71d3f975a609be174dad004e9f40060de52254))
* **web:** PUT /api/spec/tables/{t} — row_count edit (S-121) ([5ee5c56](https://github.com/dbsprout/dbsprout/commit/5ee5c56788d186ffea6688cafda724138e0e09be))
* **web:** PUT /api/spec/tables/{t}/columns/{c} — edit GeneratorConfig (S-119) ([fdd09ec](https://github.com/dbsprout/dbsprout/commit/fdd09ec887d020752ed58608d8575d90353ff02b))
* **web:** PUT /api/spec/tables/{t}/columns/{c} — edit GeneratorConfig (S-119) ([5294f76](https://github.com/dbsprout/dbsprout/commit/5294f76678a0005622882ab10e4bd6075065f62a))
* **web:** seed control UI + reproducibility wiring (S-127) ([dd60e9c](https://github.com/dbsprout/dbsprout/commit/dd60e9ca8f458ae21a1d81b579d4522035f2afda))
* **web:** seed control UI + reproducibility wiring (S-127) ([c97abae](https://github.com/dbsprout/dbsprout/commit/c97abae8f29ef2cfad747df119c4845e4330e0c3))
* **web:** Studio shell layout with 4 named-slot panels (S-117) ([4f67df5](https://github.com/dbsprout/dbsprout/commit/4f67df5fa74f623b9a781e9d643fcbee7ced5d7a))
* **web:** Studio shell layout with 4 named-slot panels (S-117) ([9ec76d1](https://github.com/dbsprout/dbsprout/commit/9ec76d1c96718e3e73c9d2b2784b162504828301))
* **web:** violation row → studio:focus-cell drill-down (S-135) ([b29054b](https://github.com/dbsprout/dbsprout/commit/b29054b201fee33843727de54882680f0a64f428))
* **web:** violation row drill-down → studio:focus-cell (S-135) ([73066f9](https://github.com/dbsprout/dbsprout/commit/73066f9882f996e4447e257e8ca25161cf527924))
* **web:** wire ProgressHub + register /ws/jobs route (S-109) ([a7b0894](https://github.com/dbsprout/dbsprout/commit/a7b08948ea6337953ffca968c57a747538925822))
* **web:** wire wizard steps over real backend with step gating (S-143) ([d307679](https://github.com/dbsprout/dbsprout/commit/d307679906c5f3428e192be9bcef8c028012c326))
* **web:** wire wizard steps over real backend with step gating (S-143) ([1022145](https://github.com/dbsprout/dbsprout/commit/1022145506b46e54824649c6802217fe5ba3c4a1))
* **web:** wire Workspace onto app.state in create_app (S-111) ([caa075a](https://github.com/dbsprout/dbsprout/commit/caa075a2d4b6b04bc362a9f041704aa445279a0c))
* **web:** wizard shell + Workspace state (S-142) ([707537b](https://github.com/dbsprout/dbsprout/commit/707537b6240ee27947e5a0683dc62d3a51dc4ed7))
* **web:** wizard shell + Workspace state (S-142) ([e56fe04](https://github.com/dbsprout/dbsprout/commit/e56fe0449e9610c4e84a74b15efe07d63c631506))
* **web:** write-guard preview + HMAC token validation (S-137) ([11e61ab](https://github.com/dbsprout/dbsprout/commit/11e61ab707fbcdf396f8c3d95844e3574e25705f))
* **web:** write-guard preview + HMAC token validation (S-137) ([68043d3](https://github.com/dbsprout/dbsprout/commit/68043d3c289086f4f42864421a1ec96e82cbea83))


### Bug Fixes

* **config:** omit dbsprout/tui/* from coverage and add textual to mypy ignore ([b64ce58](https://github.com/dbsprout/dbsprout/commit/b64ce580f18533b9264fdb4df3bf41502ca288e6))
* **DBS-112:** reject non-regular-file schema inputs before reading [S-095] ([e45deae](https://github.com/dbsprout/dbsprout/commit/e45deae01ec0b6c04f20a5e9ef03652ee50abe39))
* **generate:** enforce single-column PK uniqueness via table.primary_key ([e6451ee](https://github.com/dbsprout/dbsprout/commit/e6451ee1372ffe31c9fdf46865704efc0d3d09ac))
* **generate:** enforce single-column PK uniqueness via table.primary_key ([c73ae0b](https://github.com/dbsprout/dbsprout/commit/c73ae0b484947dac2d548fba5b17044a7b6807c4))
* **mypy:** add dbsprout.tui.* ignore_errors override for environments where ([b07fb52](https://github.com/dbsprout/dbsprout/commit/b07fb526d6db309525e5fb56e16f0cb1829d5732))
* **quality:** add explicit numpy type params to detection.py [S-101] ([15c497c](https://github.com/dbsprout/dbsprout/commit/15c497c4e3a43ab974487104c14c3036ba39d844))
* **quality:** add explicit numpy type params to detection.py [S-101] ([40b4136](https://github.com/dbsprout/dbsprout/commit/40b4136c4c83126a4e3d88e5d8ff934a8822c055))
* **S-098:** preserve dialect detection + ENUM raw_type; de-flake perf tests ([a0b8f88](https://github.com/dbsprout/dbsprout/commit/a0b8f880612da5c4743c4940aabcafd6a7f68c55))
* **web:** drop unsupported erDiagram click directives; bind clicks in JS ([71002bc](https://github.com/dbsprout/dbsprout/commit/71002bc4e3ad5a0f0d4d8e090485e83c1773a63d))
* **web:** drop unsupported erDiagram click directives; bind clicks in JS ([1464a86](https://github.com/dbsprout/dbsprout/commit/1464a86ec3fb570b556f61ca2df0670a100ae394))
* **web:** restore /api/schema/erd after _build_erd_with_clicks rename ([6757c1b](https://github.com/dbsprout/dbsprout/commit/6757c1b20844ed658a5e336a6c7f1c797b62d285))


### Performance

* **S-098:** optimise parse_ddl — ~2.9x speedup for large DDL files ([03f0418](https://github.com/dbsprout/dbsprout/commit/03f04187ad32b5de1e2931e62a5cb0e4f051fc1f))
* **S-098:** optimise parse_ddl — ~2.9x speedup for large DDL files ([61a4737](https://github.com/dbsprout/dbsprout/commit/61a47379c03559db0eefb9fb64e2425334e2fcf2))


### Code Refactoring

* **core:** add service facade; route CLI generate/validate through it (S-106) ([c459b86](https://github.com/dbsprout/dbsprout/commit/c459b866fe873e9ee9b52323360f3e3504a3543d))
* **core:** add service facade; route CLI generate/validate through it (S-106) ([509aeb7](https://github.com/dbsprout/dbsprout/commit/509aeb7f8d196ff07b3ab1b035319a941e7002e7))
* **DBS-140:** extract CLI direct-insert adapter from generate.py [S-106-F1] ([dbe3094](https://github.com/dbsprout/dbsprout/commit/dbe3094e29e890faa875fdfb6643f9bab7f827cd))
* **DBS-140:** extract CLI direct-insert adapter from generate.py [S-106-F1] ([a08998c](https://github.com/dbsprout/dbsprout/commit/a08998c491ff602cfe77c599b73debd20eed4e92))
* **DBS-95:** drop needless jinja raw block in schema template [S-091] ([2517e2d](https://github.com/dbsprout/dbsprout/commit/2517e2d274223535016fc8769911563ae084e441))
* **generate:** per-column random.Random in HeuristicEngine [S-104] ([4133ac6](https://github.com/dbsprout/dbsprout/commit/4133ac6eca46a4c4e9d72885c4ceaa9b2f2d71d3))
* **generate:** per-column random.Random in HeuristicEngine [S-104] ([a478cbf](https://github.com/dbsprout/dbsprout/commit/a478cbf48b2a0b4eb9688143c11943a64cc3b69c))
* **train:** decompose QLoRATrainer._run_unsloth under 50-line limit (S-099) ([76f3c92](https://github.com/dbsprout/dbsprout/commit/76f3c92648f1c71b565bc9ed5b339d627b80e97c))
* **train:** decompose QLoRATrainer._run_unsloth under 50-line limit (S-099) ([51c1ed0](https://github.com/dbsprout/dbsprout/commit/51c1ed0233a9b4e79b24b11531e3ece8ee4a4bab))
* **web:** harden schema-load parse against unexpected errors (S-113) ([16bf122](https://github.com/dbsprout/dbsprout/commit/16bf12202e5763388cda4c5a258d62e26437e112))
* **web:** JSON-canonical scope hash for update-column tokens ([e302d25](https://github.com/dbsprout/dbsprout/commit/e302d25997ec55b4c6d85e7eb5fb7f182c86420b))
* **web:** redacting __repr__ + validation caveat on Workspace (S-111) ([a307bfd](https://github.com/dbsprout/dbsprout/commit/a307bfd3e3f610da2f63c88aba557511a8c352ad))


### Miscellaneous

* **ci:** fix pre-existing mypy and coverage gate failures [S-104] ([19392fb](https://github.com/dbsprout/dbsprout/commit/19392fb224a23943decd0ace920faefe8f4a6213))
* **S-091-F1:** restore dev coverage/mypy config (drop out-of-scope edits) ([6ecc632](https://github.com/dbsprout/dbsprout/commit/6ecc63254592b44867ba9e4e6cb40025765efdd1))
* **S-104:** restore dev coverage/mypy config (drop out-of-scope edits) ([671243a](https://github.com/dbsprout/dbsprout/commit/671243aeb3867aa56fe31a1c4a8f449942c712f6))
* **spec:** silence bandit B105 false-positives on description strings (S-123) ([24697a4](https://github.com/dbsprout/dbsprout/commit/24697a4996e4e33d8c312e8af14b62fcb7a81ec9))


### CI/CD

* **DBS-116:** make benchmark regression gate advisory [S-096] ([4387259](https://github.com/dbsprout/dbsprout/commit/4387259ed9437f074d9d22c737bb21d96d276265))
* **DBS-116:** move advisory flag to regression step so the check stays green [S-096] ([dae8d47](https://github.com/dbsprout/dbsprout/commit/dae8d47b0ef5c9e0d3f364c8ece04f7de1367009))
* install [tui] extra in lint+test jobs for Textual TUI ([f2ec67b](https://github.com/dbsprout/dbsprout/commit/f2ec67b4767df4da43d409d98ad4bd44979a216f))


### Tests

* **output:** cover safe-ident + writer guards -&gt; 100% (S-138) ([f6ebe6c](https://github.com/dbsprout/dbsprout/commit/f6ebe6c2493d75d7d9f7e6bd953dac7ecef71c51))
* **web:** assert generate route registered + lazy-import contract (S-124) ([4f35cbe](https://github.com/dbsprout/dbsprout/commit/4f35cbe5b033cf8359b0bf51d5619170c4b8be1d))
* **web:** bound the cancel-test busy-wait so a regression fails fast (S-108) ([c616809](https://github.com/dbsprout/dbsprout/commit/c616809dfe9aa040f4fe02a5b91d395481e2a3a6))
* **web:** cover _scrub credential-helper branches to 100% (S-124) ([495633a](https://github.com/dbsprout/dbsprout/commit/495633a74b650f7d8ed7ef5b0e52490dc50e461b))
* **web:** cover _wants_html missing-Accept-header branch (S-118) ([c94ccd1](https://github.com/dbsprout/dbsprout/commit/c94ccd15a0b56d5597e504817c797496fd1c9b5f))
* **web:** cover live-tail + disconnect paths for progress WS (S-109) ([319d1a9](https://github.com/dbsprout/dbsprout/commit/319d1a97d0f0109cce4ebb2fafc7ba9964fccfa1))
* **web:** cover Workspace redaction fallback + mask helpers (S-111) ([1ef305f](https://github.com/dbsprout/dbsprout/commit/1ef305f4c01437c2449a9150714ee69e79b433f2))
* **web:** generate job runs to completion + sets workspace.last_result (S-124) ([6e19b13](https://github.com/dbsprout/dbsprout/commit/6e19b13974fa874620972d5cd0164fcc7c8d1ba5))
* **web:** lock no-schema 4xx + request validation for generate (S-124) ([1fb60c9](https://github.com/dbsprout/dbsprout/commit/1fb60c914b615f5b619b4e88f3fe32932936f394))
* **web:** RED for ProgressHub + /ws/jobs WebSocket (S-109) ([3866403](https://github.com/dbsprout/dbsprout/commit/386640332b751ed773ce5a88df771bf1467b9b90))
* **web:** RED tests for in-memory Workspace session (S-111) ([12b4d23](https://github.com/dbsprout/dbsprout/commit/12b4d233bea817a951008530850d23c76484ca5f))
* **web:** redact creds on job failure + 409 on second submit (S-124) ([2c29e9a](https://github.com/dbsprout/dbsprout/commit/2c29e9a4ada2c5b9150078d35be4d9b0979156e1))
* **web:** scaffold generate router + request model (S-124) ([474fd0a](https://github.com/dbsprout/dbsprout/commit/474fd0ae62a04149802fab83882724806081d55c))

## [0.1.7](https://github.com/dbsprout/dbsprout/compare/v0.1.6...v0.1.7) (2026-05-18)


### Features

* **cli:** --report flag + standalone report command [S-085] ([af00e4d](https://github.com/dbsprout/dbsprout/commit/af00e4dcb36773387606060b93ad12292f32bfd6))
* **cli:** --report flag + standalone report command [S-085] ([e9f1be2](https://github.com/dbsprout/dbsprout/commit/e9f1be23087957023048dda6a0f5e0bc9c645eff))
* **DBS-100:** per-table progress bar for PgCopyWriter direct insertion [S-042-F1] ([92e4663](https://github.com/dbsprout/dbsprout/commit/92e46638642dd6af218c0739791b6180c88eb922))
* **DBS-100:** per-table progress bar for PgCopyWriter direct insertion [S-042-F1] ([1956a39](https://github.com/dbsprout/dbsprout/commit/1956a399d5ca481bb900f631ee89b1791790bb5e))
* **DBS-102:** configurable max_rows_per_table guard in orchestrator [S-094] ([1516e73](https://github.com/dbsprout/dbsprout/commit/1516e7311bdf2a2509e772b943a1191dcdbf2dd5))
* **DBS-102:** consistent quoted install-hint format across writers [S-094] ([ec44495](https://github.com/dbsprout/dbsprout/commit/ec44495cbfca235cfa9b857ae0d0e52cb2f2e497))
* **DBS-102:** MySQL _quote_string escapes NUL/LF/CR/Ctrl-Z [S-094] ([ec038c5](https://github.com/dbsprout/dbsprout/commit/ec038c5c888e40eeec0a279a002be5b33a1d54b9))
* **DBS-102:** output writer hardening — security, error handling, forward compat [S-094] ([2f790c3](https://github.com/dbsprout/dbsprout/commit/2f790c3b372ecdc0f576cfa539853b56bf215333))
* **DBS-102:** parquet pl.String migration + tz-aware datetimes + snappy test [S-094] ([0a31395](https://github.com/dbsprout/dbsprout/commit/0a31395fba70b387f7710cc018ca2e71e3ce4cb7))
* **DBS-102:** restrict parquet + mysql temp file perms to 0o640 [S-094] ([770b71e](https://github.com/dbsprout/dbsprout/commit/770b71eb0738dc8ddfcdf883422ad918e1886ca0))
* **DBS-102:** restrict sql/csv/json output file perms to 0o640 [S-094] ([b6001ba](https://github.com/dbsprout/dbsprout/commit/b6001ba026dd25a7753fd56a2c25c0207a032e0c))
* **DBS-102:** scrub cell values from Parquet build errors + smallint overflow msg [S-094] ([0090cbc](https://github.com/dbsprout/dbsprout/commit/0090cbc773db370d995f58f65c1c696b4464bc95))
* **DBS-102:** shared output file-permission helper [S-094] ([76cf33c](https://github.com/dbsprout/dbsprout/commit/76cf33c65f43f0b190ea71e2e893004b6a41d7a1))
* **DBS-102:** validate pk_columns subset in build_upsert [S-094] ([52f5938](https://github.com/dbsprout/dbsprout/commit/52f59383a131b99abbbdb12e3047589ddf130a71))
* **DBS-102:** warn on --upsert with non-SQL format; document SQLite PK order [S-094] ([9b4bacb](https://github.com/dbsprout/dbsprout/commit/9b4bacb956c10367c3309b2f78b02ff8354488bb))
* **DBS-105:** extend control-char regex to C1 range [S-002b] ([df38daa](https://github.com/dbsprout/dbsprout/commit/df38daa505a7877aa725d6e000edb34dc4fed43f))
* **DBS-105:** FK field Identifier validation and DDL defense-in-depth [S-002B] ([3c1bd56](https://github.com/dbsprout/dbsprout/commit/3c1bd56d0f1c16d456f106a39aeda11eb0ac9354))
* **DBS-105:** Identifier-validate FK/Index columns and ref_table [S-002b] ([c9feaba](https://github.com/dbsprout/dbsprout/commit/c9feabaeff1e39efe0df48805f785f2cb8365a38))
* **DBS-105:** validate ForeignKeySchema.initially as DEFERRED|IMMEDIATE [S-002b] ([06fbc75](https://github.com/dbsprout/dbsprout/commit/06fbc7575eb711225b434b11b2a97a34bec048cf))
* **DBS-107:** harden dbsprout diff path/snapshot/markup input validation [S-054a] ([a053ff1](https://github.com/dbsprout/dbsprout/commit/a053ff1405eb77ba9db01f21b3704a1a9e1d3e96))
* **DBS-107:** symlink/traversal guard + sanitized parse errors on diff --file [S-054a] ([53a165a](https://github.com/dbsprout/dbsprout/commit/53a165a33208e2ab5abeb345c45d59acf6b4c63e))
* **DBS-107:** validate --snapshot hex prefix + escape not-found message [S-054a] ([fde5cea](https://github.com/dbsprout/dbsprout/commit/fde5cea2eea5c63455ec8f7b0abaeeaa1fe10ba8))
* **DBS-115:** wire generate --lora flag [S-067b] ([1082399](https://github.com/dbsprout/dbsprout/commit/1082399b5abd29c090494d450287508bbb799879))
* **DBS-115:** wire generate --lora flag end-to-end [S-067b] ([ffc4951](https://github.com/dbsprout/dbsprout/commit/ffc49510ff2e6cf90224f80f9a3a5d8d2c1cb769))
* **DBS-126:** direct-insertion UPSERT via staging tables [S-094-F1] ([c527b53](https://github.com/dbsprout/dbsprout/commit/c527b53acc8968c949a11417043e3ddf466926c1))
* **DBS-126:** direct-insertion UPSERT via staging tables [S-094-F1] ([ab3200f](https://github.com/dbsprout/dbsprout/commit/ab3200f4dccf7d2faa0fb39315d709a0b180a55e))
* **DBS-55:** schema snapshot storage in .dbsprout/snapshots/ [S-051] ([#68](https://github.com/dbsprout/dbsprout/issues/68)) ([3981d6d](https://github.com/dbsprout/dbsprout/commit/3981d6d80cc89462d181e5a5ee9c97fc227bddaa))
* **DBS-56:** schema diff algorithm compare old/new to list[SchemaChange] [S-052] ([#69](https://github.com/dbsprout/dbsprout/issues/69)) ([ae9ec5a](https://github.com/dbsprout/dbsprout/commit/ae9ec5ae942bf4541145c09515c7d4d7d0c331f4))
* **DBS-57:** incremental seed updater with per-change-type update rules [S-053] ([#70](https://github.com/dbsprout/dbsprout/issues/70)) ([36521cb](https://github.com/dbsprout/dbsprout/commit/36521cb9ca892c4691f9c66bca8b1031e2cb7be3))
* **DBS-58:** dbsprout diff command for human-readable schema drift report [S-054] ([#71](https://github.com/dbsprout/dbsprout/issues/71)) ([67b74d9](https://github.com/dbsprout/dbsprout/commit/67b74d90ce414aee681a05aa10fdfb198cacbdcf))
* **DBS-59:** --incremental flag on generate command [S-055] ([#72](https://github.com/dbsprout/dbsprout/issues/72)) ([b61ea37](https://github.com/dbsprout/dbsprout/commit/b61ea3770d501c7a0d955e40c04b208679e505cd))
* **DBS-60:** Alembic migration parser [S-056] ([#73](https://github.com/dbsprout/dbsprout/issues/73)) ([bd8c995](https://github.com/dbsprout/dbsprout/commit/bd8c9953c9de99fc05fa9376c222f892fa4f1356))
* **DBS-61:** Django migration parser [S-057] ([#75](https://github.com/dbsprout/dbsprout/issues/75)) ([8c9803a](https://github.com/dbsprout/dbsprout/commit/8c9803a190ad28cf7b1c38188c49436d253af062))
* **DBS-62:** Flyway SQL migration parser [S-058] ([#76](https://github.com/dbsprout/dbsprout/issues/76)) ([8db55fe](https://github.com/dbsprout/dbsprout/commit/8db55fe2ecca492ea1b7e8d38cf026a2e1a25d39))
* **DBS-63:** Liquibase XML changelog parser [S-059] ([#77](https://github.com/dbsprout/dbsprout/issues/77)) ([b5b41e4](https://github.com/dbsprout/dbsprout/commit/b5b41e41cf830ec703711fe755cf94985e2cdd44))
* **DBS-64:** Prisma migration parser [S-060] ([#78](https://github.com/dbsprout/dbsprout/issues/78)) ([f32c2c3](https://github.com/dbsprout/dbsprout/commit/f32c2c30c80fb7d4ac8191f252ad4e14144b7c76))
* **DBS-65:** plugin entry-point system [S-061] ([#79](https://github.com/dbsprout/dbsprout/issues/79)) ([cc29f30](https://github.com/dbsprout/dbsprout/commit/cc29f30b18605a54b839621f12756e81a0d8f072))
* **DBS-66:** sample extraction stratified sampling [S-062] ([#80](https://github.com/dbsprout/dbsprout/issues/80)) ([eda1b83](https://github.com/dbsprout/dbsprout/commit/eda1b83b231da7de1e198455e06e1d28bbce5b23))
* **DBS-67:** GReaT-style row serialization to JSONL [S-063] ([#81](https://github.com/dbsprout/dbsprout/issues/81)) ([12bc8e8](https://github.com/dbsprout/dbsprout/commit/12bc8e8c69991abfefd79b3dc0c90cc89e79b69d))
* **DBS-68:** QLoRA trainer with CUDA auto-detect to Unsloth [S-064] ([afdac9d](https://github.com/dbsprout/dbsprout/commit/afdac9db30ec5eb63fccaa3af676b92e7a645a70))
* **DBS-68:** QLoRA trainer with CUDA auto-detect to Unsloth [S-064] ([eb249fc](https://github.com/dbsprout/dbsprout/commit/eb249fc6d09fa0dfef6e620033f17ac90ae306f0))
* **DBS-69:** MLX trainer for Apple Silicon path [S-065] ([bb81ddf](https://github.com/dbsprout/dbsprout/commit/bb81ddfe4e71893a14b92c2fcea7457fdcbdf558))
* **DBS-69:** MLX trainer for Apple Silicon path [S-065] ([1a760b6](https://github.com/dbsprout/dbsprout/commit/1a760b6a140b2b59dafe6a1b890df77a2de4133a))
* **DBS-70:** GGUF export merge LoRA to safetensors to Q4_K_M [S-066] ([0ceb9f1](https://github.com/dbsprout/dbsprout/commit/0ceb9f1458b1beb960f53cff890ed01a22f69ab8))
* **DBS-70:** GGUF export merge LoRA to safetensors to Q4_K_M [S-066] ([1a5b311](https://github.com/dbsprout/dbsprout/commit/1a5b31198eca84fed65fdda4b9d9760cabe011ea))
* **DBS-71:** LoRA hot-swap adapter loading without restart [S-067] ([5d06f07](https://github.com/dbsprout/dbsprout/commit/5d06f0750e916c67f48c0d85979b9dcff5186d06))
* **DBS-71:** LoRA hot-swap adapter loading without restart [S-067] ([c9631b7](https://github.com/dbsprout/dbsprout/commit/c9631b77a3e0c9e508bd3cdb0b1310ec5dc46fdf))
* **DBS-72:** dbsprout train end-to-end pipeline command [S-068] ([dea14a9](https://github.com/dbsprout/dbsprout/commit/dea14a9a55fb099c639d971f04b46b53d3e74fa4))
* **DBS-72:** dbsprout train end-to-end pipeline command [S-068] ([3e455bc](https://github.com/dbsprout/dbsprout/commit/3e455bce155bd86a43d49ec6d71aba30c130f87e))
* **DBS-73:** bundled model registry manifest + loader [S-069] ([17468c8](https://github.com/dbsprout/dbsprout/commit/17468c8e2b257eab43f6290c2f69edee2ce7da35))
* **DBS-73:** dbsprout models download with Rich progress + resume [S-069] ([3efad9f](https://github.com/dbsprout/dbsprout/commit/3efad9f0a08827c1b174ddb61d0b094155951c5c))
* **DBS-73:** dbsprout models list/download/info command [S-069] ([fa0738d](https://github.com/dbsprout/dbsprout/commit/fa0738d896531648d5c3f04c82c073317f76c5c4))
* **DBS-73:** dbsprout models list/info Typer commands [S-069] ([f847f74](https://github.com/dbsprout/dbsprout/commit/f847f748e2c6039d991f60f793d5ce8b09f4c89c))
* **DBS-73:** ModelManager discovery and install-status [S-069] ([01b4492](https://github.com/dbsprout/dbsprout/commit/01b44923345177cb7d0e08dc3c8cc600c10ad65f))
* **DBS-73:** models package with frozen Pydantic types [S-069] ([1c25bcb](https://github.com/dbsprout/dbsprout/commit/1c25bcb0db7635d7a5800705bb9d5800a11c8f51))
* **DBS-73:** resumable httpx model download with progress callback [S-069] ([20bff8a](https://github.com/dbsprout/dbsprout/commit/20bff8a6841f75face9608bbda63dad9223ed71c))
* **DBS-74:** training privacy — PII redaction, target masking, epoch cap, DP-SGD guard [S-070] ([c8f4826](https://github.com/dbsprout/dbsprout/commit/c8f48266a7cfda82dde485dbf0428f22e49dadf3))
* **DBS-74:** training-privacy module — config, stats, dp_sgd guard, redactor [S-070] ([3573d41](https://github.com/dbsprout/dbsprout/commit/3573d4146274b19d2fa6ba0e1ae7b97cbda8bbf4))
* **DBS-74:** wire PII redaction + --no-pii-redaction + DP-SGD guard into pipeline [S-070] ([a2e655c](https://github.com/dbsprout/dbsprout/commit/a2e655c46127b1d41760fa4f8d9882ef68b765c6))
* **DBS-75:** GaussianCopula statistical generation engine [S-071] ([d59ed67](https://github.com/dbsprout/dbsprout/commit/d59ed67ea8d0913764149e5c70fd86a2675ec88b))
* **DBS-75:** GaussianCopula statistical generation engine [S-071] ([10aac34](https://github.com/dbsprout/dbsprout/commit/10aac34c0f151d6638419629552f53594f90e11a))
* **DBS-76:** MongoDB schema inference via pymongo-schema [S-072] ([dce02e2](https://github.com/dbsprout/dbsprout/commit/dce02e2ffd2ce16acdd05b261ac15dce5a4117ab))
* **DBS-76:** MongoDB schema inference via pymongo-schema [S-072] ([ef46c7b](https://github.com/dbsprout/dbsprout/commit/ef46c7b2752607422812eced4b8e7fb6df7f9a2a))
* **DBS-77:** CheckResult model + doctor package skeleton [S-073] ([e5db621](https://github.com/dbsprout/dbsprout/commit/e5db6219daa465fc5ca561e1f879830f4358be54))
* **DBS-77:** dbsprout doctor environment diagnostics command [S-073] ([6d6bc54](https://github.com/dbsprout/dbsprout/commit/6d6bc54a73f6f0257f7f914934dfadd0bf882248))
* **DBS-77:** doctor CLI command module + Rich rendering [S-073] ([ca0797c](https://github.com/dbsprout/dbsprout/commit/ca0797cf697492d859a7e65f819d1acafea85f80))
* **DBS-77:** doctor database-connectivity check [S-073] ([5378d6a](https://github.com/dbsprout/dbsprout/commit/5378d6ac274bccfa16e48ded2e76aabdb5c5f3d1))
* **DBS-77:** doctor model + disk-space checks [S-073] ([0346ace](https://github.com/dbsprout/dbsprout/commit/0346acef46697ddad43af727772fb8ef9ee24e86))
* **DBS-77:** doctor optional-extras check [S-073] ([89cc537](https://github.com/dbsprout/dbsprout/commit/89cc537da1889d4d060ce0235ccb8575fa55ff04))
* **DBS-77:** doctor plugin + secret-scan checks [S-073] ([6bde3a0](https://github.com/dbsprout/dbsprout/commit/6bde3a0ff53d5e2ae47e74f6b836686faf74be75))
* **DBS-77:** doctor Python-version check [S-073] ([314bdef](https://github.com/dbsprout/dbsprout/commit/314bdefd692269431edb4d5741ed511992a9884e))
* **DBS-77:** doctor run_all_checks orchestrator [S-073] ([bb65124](https://github.com/dbsprout/dbsprout/commit/bb6512433bc36c5cf497ec649a1e23a2e5a5f52c))
* **DBS-77:** doctor training-accelerator check [S-073] ([27fc1c2](https://github.com/dbsprout/dbsprout/commit/27fc1c2ea4efdeabb8dfaf8d98ae8fa1c89e357e))
* **DBS-77:** wire dbsprout doctor command into CLI [S-073] ([74f0cd0](https://github.com/dbsprout/dbsprout/commit/74f0cd0051d4abe447a192810a43974a9680bd13))
* **DBS-78:** performance benchmark suite with pytest-benchmark [S-074] ([ffe6230](https://github.com/dbsprout/dbsprout/commit/ffe62307505bc493a00f346058c90609a15d6605))
* **DBS-78:** real YAML schema parser in example plugin [S-078] ([120c6ca](https://github.com/dbsprout/dbsprout/commit/120c6ca0ec762c4be664430bd7b4122267905b10))
* **DBS-80:** add DBSprout error hierarchy + MissingDependencyError [S-076] ([9bdc62c](https://github.com/dbsprout/dbsprout/commit/9bdc62c0c61f30730afc54c3de5ab0ddd7e4b7cd))
* **DBS-80:** global --verbose flag + run() error-guard entrypoint [S-076] ([e6247a5](https://github.com/dbsprout/dbsprout/commit/e6247a53e6fab02ae53857d8205b7d7a60fb945e))
* **DBS-80:** Rich what/why/fix error panel + CLI guard [S-076] ([4e8b158](https://github.com/dbsprout/dbsprout/commit/4e8b15860f02c16408b4c74f1c7504c9c27ffcff))
* **DBS-80:** user-friendly what/why/fix error handling + missing-dependency hints [S-076] ([6411de0](https://github.com/dbsprout/dbsprout/commit/6411de0ba153988b6abca6ebf598e96609cdc394))
* **DBS-81:** mkdocs-material scaffold + landing page [S-077] ([be4bee2](https://github.com/dbsprout/dbsprout/commit/be4bee2d37513e45874715035fea57d9e766ac39))
* **DBS-82:** plugin docs + working YAML example plugin [S-078] ([ae59a32](https://github.com/dbsprout/dbsprout/commit/ae59a322dd261b9848020b5579cc15992f83a643))
* **DBS-83:** SQLite state layer schema [S-079] ([4c82909](https://github.com/dbsprout/dbsprout/commit/4c8290916dc2ce1783893a3bf023f74f5531bf0b))
* **DBS-83:** SQLite state layer schema [S-079] ([8d09ec8](https://github.com/dbsprout/dbsprout/commit/8d09ec8fd5be9380a0d90148ae0797157f5b58d5))
* **DBS-S067c:** --lora overrides [llm].lora_path precedence [S-067c] ([a62ac62](https://github.com/dbsprout/dbsprout/commit/a62ac623f2f0b1b266fac3b07b7126a7c352eb67))
* **DBS-S067c:** [llm] TOML config section + --lora config precedence [S-067c] ([6aa48e2](https://github.com/dbsprout/dbsprout/commit/6aa48e2223c55efc76925ad5fa3750abc3a3a6ad))
* **DBS-S067c:** add [llm] LLMConfig section + TOML round-trip [S-067c] ([c380dcb](https://github.com/dbsprout/dbsprout/commit/c380dcb6b6db73dbf7617e891a79c65c24136764))
* **DBS-S097:** _make_private Opacus seam (epsilon/noise modes) [S-097] ([b2193d2](https://github.com/dbsprout/dbsprout/commit/b2193d297bd4e26ab986973fa45f88038de7abbe))
* **DBS-S097:** add train-dp (opacus) optional extra + lockfile [S-097] ([eead3c2](https://github.com/dbsprout/dbsprout/commit/eead3c2a9fcd3f5c304b5f7112462206dee8ea38))
* **DBS-S097:** full DP-SGD (Opacus) integration into training backend [S-097] ([60cff62](https://github.com/dbsprout/dbsprout/commit/60cff62024896b3210e48a4651ce47f4ebe11d09))
* **DBS-S097:** LoRAAdapter achieved_epsilon/dp_delta fields [S-097] ([54f6e16](https://github.com/dbsprout/dbsprout/commit/54f6e16dce128cf6a1e8d556fc17890dd5a7faeb))
* **DBS-S097:** MLX backend rejects DP-SGD with clear not-supported error [S-097] ([b176336](https://github.com/dbsprout/dbsprout/commit/b1763361a1930f25f85adb204d6595614c8083ef))
* **DBS-S097:** relax dp_sgd_guard to only raise when Opacus absent [S-097] ([717f10d](https://github.com/dbsprout/dbsprout/commit/717f10de40b29dd77d8605dc13e7f2795d27fcca))
* **DBS-S097:** surface (epsilon, delta) in train pipeline summary [S-097] ([e41e595](https://github.com/dbsprout/dbsprout/commit/e41e5954976383a7467b4a377a64096eda22d634))
* **DBS-S097:** TrainPrivacyConfig DP hyperparameters + exactly-one validator [S-097] ([a02f49b](https://github.com/dbsprout/dbsprout/commit/a02f49b29f7b0fc9ecaa612032a3deefa9fbb5f2))
* **DBS-S097:** wire Opacus DP-SGD into QLoRATrainer + thread epsilon [S-097] ([603064e](https://github.com/dbsprout/dbsprout/commit/603064ed935afb9b71105496aa023fb59030e66c))
* **report:** categorical value-frequency bar specs [S-083] ([7cf858c](https://github.com/dbsprout/dbsprout/commit/7cf858cbe87a70534c15368ad691b3de8b07252a))
* **report:** chart + status-badge styles; 100% report cov [S-083] ([857d787](https://github.com/dbsprout/dbsprout/commit/857d787b83392155e6dc1cf9a4691af7f19257d0))
* **report:** chart-bundle aggregator [S-083] ([0e1a10c](https://github.com/dbsprout/dbsprout/commit/0e1a10ca003d32c712b9aa713bd2cd331c7c1331))
* **report:** classified quality-metrics table partial [S-083] ([6f810bb](https://github.com/dbsprout/dbsprout/commit/6f810bb2d1fad82ef05358e6326bd1727ba76ac5))
* **report:** correlation heatmap spec [S-083] ([fcf1615](https://github.com/dbsprout/dbsprout/commit/fcf1615916923e4d6e4e244eb6c203ba452d317a))
* **report:** data preview tables in HTML report [S-084] ([e79d033](https://github.com/dbsprout/dbsprout/commit/e79d0332429593fbcf12c60ee8681f4d2a643e18))
* **report:** data preview tables in HTML report [S-084] ([9bd98e5](https://github.com/dbsprout/dbsprout/commit/9bd98e5399e6297b2771ba18b89a33c18926f359))
* **report:** expose charts+quality_table in context [S-083] ([bde2b23](https://github.com/dbsprout/dbsprout/commit/bde2b234d1519b29a9e706f5d8459c867fe80f5c))
* **report:** Mermaid ERD diagram in HTML report [S-082] ([0fc7ece](https://github.com/dbsprout/dbsprout/commit/0fc7ece8fed9bf0a1ca8ea003567f85941cafee1))
* **report:** Mermaid ERD diagram in HTML report [S-082] ([d4cebe6](https://github.com/dbsprout/dbsprout/commit/d4cebe6ecf56c1537938dcdf193bef4e6e8377eb))
* **report:** numeric distribution histogram specs [S-083] ([5747699](https://github.com/dbsprout/dbsprout/commit/57476994605b3586d8709ccac6351c3bffb53555))
* **report:** Plotly charts + quality metrics table [S-083] ([2103322](https://github.com/dbsprout/dbsprout/commit/2103322414a1142ae156d4c306b1b4a4a6631ea1))
* **report:** Plotly charts partial w/ embedded JSON [S-083] ([985d5f1](https://github.com/dbsprout/dbsprout/commit/985d5f1a9234dcf92366268c17acf9a4317e12ed))
* **report:** quality-table view-model w/ status [S-083] ([fadf687](https://github.com/dbsprout/dbsprout/commit/fadf687ed6378b5d63454f0a9c7ce8efe67c7bc2))
* **report:** self-contained HTML report generator [S-081] ([cd55f85](https://github.com/dbsprout/dbsprout/commit/cd55f8570a3fed7b37c4d4ba9c38644f690fd9f5))
* **report:** self-contained HTML report generator [S-081] ([c6d2e64](https://github.com/dbsprout/dbsprout/commit/c6d2e64480508bbde1dfa2c2979e19228d6f3f75))
* **spec:** real LLM token/cost accounting in state RunRecord [S-080a] ([517b286](https://github.com/dbsprout/dbsprout/commit/517b28638c6b6ad9b8dc16364626a3d4a0cc203e))
* **spec:** real LLM token/cost accounting in state RunRecord [S-080a] ([578d640](https://github.com/dbsprout/dbsprout/commit/578d640b784789e5465606261cc9f286ea705190))
* **state:** record run data on generate [S-080] ([9352b98](https://github.com/dbsprout/dbsprout/commit/9352b98d230eb72d9b81d394622725995859399b))
* **state:** record run data on generate [S-080] ([565adac](https://github.com/dbsprout/dbsprout/commit/565adac6de691271d0103d73c5628bf2a520d381))


### Bug Fixes

* consolidated code-review fixes for S-064..S-069 [DBS-68..DBS-73] ([04139d9](https://github.com/dbsprout/dbsprout/commit/04139d958d7f3eaa700c16196fabeea71447cb5b))
* **DBS-102:** cap + sanitize reference CSV; drop unused table_name [S-094] ([8c762ce](https://github.com/dbsprout/dbsprout/commit/8c762ceb31ca9688dbbcf561dc79b7df1fd84f70))
* **DBS-102:** clamp NaN/Inf scores; escape Rich markup in quality tables [S-094] ([6812c86](https://github.com/dbsprout/dbsprout/commit/6812c865c4f764ed8c77aafe1b1e3305d78da9bd))
* **DBS-102:** correct Frobenius bound; consistent missing-scipy error [S-094] ([5dd6013](https://github.com/dbsprout/dbsprout/commit/5dd6013a55da03b502458bec6fb544bd15b3b5b2))
* **DBS-103:** infer Django dialect from DATABASES settings [S-102] ([146fa48](https://github.com/dbsprout/dbsprout/commit/146fa480c53b806cb9fe906ad51c88f763f2d651))
* **DBS-103:** infer Django dialect from DATABASES settings [S-102] ([90a0988](https://github.com/dbsprout/dbsprout/commit/90a0988630a41e9be3dcd9115fc7f9b93732f8ed))
* **DBS-104:** robust temp-path escaping in MySQL LOAD DATA writer [S-103] ([08edba0](https://github.com/dbsprout/dbsprout/commit/08edba0f57cf383949c55233c9b79646c774de67))
* **DBS-104:** robust temp-path escaping in MySQL LOAD DATA writer [S-103] ([4e0bd67](https://github.com/dbsprout/dbsprout/commit/4e0bd67bdd392259acaf62e391bb28d07677711f))
* **DBS-106:** harden init.py TOML generation against injection [S-051b] ([31b670c](https://github.com/dbsprout/dbsprout/commit/31b670cf4678cc3c357aaed7e446d87cea86b1d4))
* **DBS-106:** harden init.py TOML generation against injection [S-051b] ([771368b](https://github.com/dbsprout/dbsprout/commit/771368b20f221259f5007c80e47082a6a40faca1))
* **DBS-106:** re-harden existing snapshot on idempotent re-save [S-051b] ([44d5ce9](https://github.com/dbsprout/dbsprout/commit/44d5ce94dbf1f3c071104343929f976d3c4d0178))
* **DBS-107:** markup-escape schema names in diff change lines [S-054a] ([97db1af](https://github.com/dbsprout/dbsprout/commit/97db1af80dbe475da3a1324a0bcd7c415ea5163e))
* **DBS-107:** skip symlinks in SnapshotStore._find_by_hash_prefix [S-054a] ([e9e6347](https://github.com/dbsprout/dbsprout/commit/e9e634746fb6802c0875bed82892a0ce230f3413))
* **DBS-113:** real mlx-lm tuner integration [S-065b] ([5186396](https://github.com/dbsprout/dbsprout/commit/51863968317147b188154ea0d986360571e89c69))
* **DBS-113:** real mlx-lm tuner train flow + adapter_config [S-065b] ([5a87291](https://github.com/dbsprout/dbsprout/commit/5a872912b844d97e6b1c844ba5e9b3eb75b7f5a5))
* **DBS-113:** seam resolves verified-real mlx-lm symbols [S-065b] ([3029428](https://github.com/dbsprout/dbsprout/commit/302942803869ee0d20b39f0a00fc4c807a484228))
* **DBS-114:** bound conversion subprocess with a timeout [S-066b] ([4de40ff](https://github.com/dbsprout/dbsprout/commit/4de40ff150d191027f740b10cfa848a1ebf365a2))
* **DBS-114:** real llama.cpp GGUF subprocess conversion [S-066b] ([4db0b80](https://github.com/dbsprout/dbsprout/commit/4db0b80c6190bb2fb6743a6dd488eaa9e4bfaff1))
* **DBS-114:** real llama.cpp GGUF subprocess conversion [S-066b] ([3e54f32](https://github.com/dbsprout/dbsprout/commit/3e54f32053d49477a4755c3e4b5cb8126ab7888d))
* **DBS-115:** make --lora help assertion ANSI/width-robust [S-067b ci] ([74c0944](https://github.com/dbsprout/dbsprout/commit/74c0944ca04af7911d469fb9a514458371f85387))
* **DBS-127:** quote bracketed install hint in MissingDependencyError [S-094-F2] ([b2f3aea](https://github.com/dbsprout/dbsprout/commit/b2f3aeaf007518d1237ee13464dc694548c59e2f))
* **DBS-127:** quote bracketed install hint in MissingDependencyError [S-094-F2] ([95102f2](https://github.com/dbsprout/dbsprout/commit/95102f280869754b527279d247f1c5c8328e3e52))
* **DBS-68:** lazy train re-exports, partial-install hint, precondition order [S-064 review] ([a888620](https://github.com/dbsprout/dbsprout/commit/a888620f849aa1a8ec0f7f2cc0e025b77d073892))
* **DBS-69:** real mlx-lm LoRA seam, _Trainer Protocol, completion-loss parity [S-065 review] ([7df7c16](https://github.com/dbsprout/dbsprout/commit/7df7c163e03f5af8202880d17d5baadd6b7895c8))
* **DBS-70:** real llama.cpp subprocess conversion, atomic output, path containment [S-066 review] ([10ab85a](https://github.com/dbsprout/dbsprout/commit/10ab85a56b609b5e9eb49ffa1a291e468ae74998))
* **DBS-71:** thread n_ctx through loader, return handle, ImportError parity [S-067 review] ([15f2876](https://github.com/dbsprout/dbsprout/commit/15f2876552eccf2188c9796afec6923c3af40dcc))
* **DBS-72:** injectable pipeline-components seam, uniform secret scrubbing [S-068 review] ([c5f9ed5](https://github.com/dbsprout/dbsprout/commit/c5f9ed58246f1299672b5b3ffcdb97bbbd7d0093))
* **DBS-73:** make httpx a core dependency so models CLI tests run in CI [S-069] ([3f8ff4d](https://github.com/dbsprout/dbsprout/commit/3f8ff4da29bef95e998fce51020c82a02ac0f3d9))
* **DBS-73:** registry path-traversal validators, sha256, list base models, redirects [S-069 review] ([b974e5b](https://github.com/dbsprout/dbsprout/commit/b974e5bbe835b03ca281f22bdbcc093f34723d3f))
* **DBS-74:** preserve non-string dtypes during PII redaction [S-070 review] ([4425d86](https://github.com/dbsprout/dbsprout/commit/4425d86a380b19589b163d4d05771486381ff6aa))
* **DBS-76:** add mongo+docs to _VALID_EXTRAS allowlist [S-072 merge] ([a4eed59](https://github.com/dbsprout/dbsprout/commit/a4eed5933816bbde92fe44cc569352b2571fe7a2))
* **DBS-78:** resolve cli.app submodule via sys.modules for py3.10 compat [S-074 ci] ([f62448d](https://github.com/dbsprout/dbsprout/commit/f62448d7edb3074289019712f78b0eef2315b345))
* **DBS-80:** scrub generic-error text, drop dead ctx.obj write [S-076 review] ([f11055b](https://github.com/dbsprout/dbsprout/commit/f11055b97c1f2e9c4c5376f08d299e311a4e9538))
* **DBS-99:** address code review findings [S-002a] ([8325e2e](https://github.com/dbsprout/dbsprout/commit/8325e2e8cf1d9c97022b20b01707e9ac74bca3ae))
* **DBS-99:** schema model input validation and DDL hardening [S-002a] ([d3c074a](https://github.com/dbsprout/dbsprout/commit/d3c074a279fcfb84395e0956b25e99ec6e965388))
* **DBS-99:** schema model input validation and DDL hardening [S-002a] ([fadfdb3](https://github.com/dbsprout/dbsprout/commit/fadfdb36c71a2e18ed33c60e65a6f1b93b26e6f5))
* **report:** reconcile gate breakage — isort-clean S-083 import region + stable charts marker in erd test [S-083] ([5e29c73](https://github.com/dbsprout/dbsprout/commit/5e29c73ccf557a649e34d3c82cf9e2b5dcdbac12))
* **S-066c:** tiny-model SPM tokenizer fixture satisfies pinned llama.cpp converter ([27ee3da](https://github.com/dbsprout/dbsprout/commit/27ee3dacbc67d25fdc6cfa2c5656132071f038b5))
* **S-097:** py3.10 tomllib import fallback in test_errors ([3547fb3](https://github.com/dbsprout/dbsprout/commit/3547fb38b4ff20e387f3b43196510e84ec57958c))
* **test:** drop spurious py3.11 guard in test_polars_has_upper_bound [S-094] ([77b531e](https://github.com/dbsprout/dbsprout/commit/77b531e30c071e6b2b5a4f38d1b64c67a6e47649))
* **test:** tomllib import fallback for Python 3.10 [S-094] ([af22efd](https://github.com/dbsprout/dbsprout/commit/af22efdb753180e9db3f9aedb89fda5e928b1830))


### Code Refactoring

* **DBS-102:** IntegrityReport/FidelityReport use tuple for true immutability [S-094] ([0b66d9c](https://github.com/dbsprout/dbsprout/commit/0b66d9c5a603d83bf5c722d1e4d3ed4c730af010))
* **DBS-110:** split test_alembic.py per-concern [S-056a] ([#74](https://github.com/dbsprout/dbsprout/issues/74)) ([8b0b008](https://github.com/dbsprout/dbsprout/commit/8b0b008652da9a3e901be257c671f08740ab7fb3))
* **DBS-113:** extract _apply_lora/_read_corpus_rows to keep _run_mlx focused [S-065b] ([e1d44c8](https://github.com/dbsprout/dbsprout/commit/e1d44c8adca82639318f2357272a00c60faeaba3))
* **DBS-115:** lazy-import spec provider in orchestrator [S-067b] ([199fb32](https://github.com/dbsprout/dbsprout/commit/199fb3210d87a74e6142cb6cf1a2a611b7f726ac))
* **DBS-80:** standardize missing-dependency install hints [S-076] ([781b59d](https://github.com/dbsprout/dbsprout/commit/781b59d84bb4c66b3df1a5c67e39e076a1496861))
* **DBS-S067c:** extract shared lora-path validation helper [S-067c] ([a77ec1d](https://github.com/dbsprout/dbsprout/commit/a77ec1d4a1710b84a8884a84e53ba990d5f2b88b))


### Documentation

* **DBS-107:** document diff --output-dir trust boundary [S-054a] ([a88bafe](https://github.com/dbsprout/dbsprout/commit/a88bafe52383b4de4a95d5c85b6c189534a2d73a))
* **DBS-78:** example plugin README [S-078] ([7020437](https://github.com/dbsprout/dbsprout/commit/70204377b8e34c430a048a548d3af68ad794cd29))
* **DBS-78:** plugin development guide [S-078] ([7cd8143](https://github.com/dbsprout/dbsprout/commit/7cd81436187e0c95303e028e7f5a240e5fe35a95))
* **DBS-81:** architecture overview + contributing; strict build green [S-077] ([4d33b1a](https://github.com/dbsprout/dbsprout/commit/4d33b1a444e252a5e8fbfcb01abbfd6f48305df8))
* **DBS-81:** CLI + configuration reference [S-077] ([abc2cc9](https://github.com/dbsprout/dbsprout/commit/abc2cc940ce70ff11f55425defacb05f8837ad6a))
* **DBS-81:** getting started (install + quick start) [S-077] ([ca16c57](https://github.com/dbsprout/dbsprout/commit/ca16c574dc51a84645f9c77aaf1770a7877ed0b1))
* **DBS-81:** mkdocs-material documentation site [S-077] ([93792cf](https://github.com/dbsprout/dbsprout/commit/93792cf45ecb06704445b5aaab2c73b63d37766b))
* **DBS-S097:** document runtime-validation-pending Opacus wiring [S-097 review] ([7a36492](https://github.com/dbsprout/dbsprout/commit/7a36492707440671127fe9b5a7063a6a9d65142b))


### Miscellaneous

* back-merge main (release-please v0.1.6 artifacts) into dev ([5c66e71](https://github.com/dbsprout/dbsprout/commit/5c66e7123ad8916624b6443ca7df987a50ee7d84))
* **DBS-102:** pin polars&lt;2.0 upper bound [S-094] ([f9c956f](https://github.com/dbsprout/dbsprout/commit/f9c956f03493de43c130ecdd7377873904f13829))
* **DBS-78:** add pytest-benchmark dep + benchmark marker [S-074] ([7bd6217](https://github.com/dbsprout/dbsprout/commit/7bd6217bc4ee544869ed18ee1e4978434e8d73c7))
* **DBS-81:** add [docs] extra and ignore site build output [S-077] ([acd0adc](https://github.com/dbsprout/dbsprout/commit/acd0adc143acdef248aa5cca3304504a7945d8f4))
* **DBS-98:** CI & supply chain hardening [S-051a] ([f48f18f](https://github.com/dbsprout/dbsprout/commit/f48f18f5af42a25c4fbb86add452e134d1c4a37a))
* **DBS-98:** CI & supply chain hardening [S-051a] ([7baec6a](https://github.com/dbsprout/dbsprout/commit/7baec6a672f2445ee13a1cd8960b2234dea7d661))


### CI/CD

* **DBS-116:** cache + provision pinned llama.cpp binary/source tarball [S-066c] ([a5c6561](https://github.com/dbsprout/dbsprout/commit/a5c6561e06287154046f2fd6d2cf26848bb55a96))
* **DBS-116:** install pinned converter deps + run exporter integration tests [S-066c] ([85fc537](https://github.com/dbsprout/dbsprout/commit/85fc5373ff54438525743a46f31e10e03ab4f20b))
* **DBS-116:** llama.cpp toolchain integration job for GGUF export test [S-066c] ([3f587d5](https://github.com/dbsprout/dbsprout/commit/3f587d5aaf19e2dc895c0c53c9ef641733261964))
* **DBS-116:** path-filtered llama.cpp integration workflow skeleton [S-066c] ([ccf0998](https://github.com/dbsprout/dbsprout/commit/ccf0998bdf9960bae584114afd61bace0bf14980))
* **DBS-78:** non-blocking benchmark job + docs [S-074] ([5cf8ec5](https://github.com/dbsprout/dbsprout/commit/5cf8ec57158a3da9f67f80b6c88182b2cf00ba99))
* **DBS-81:** GitHub Pages docs deploy workflow [S-077] ([6294953](https://github.com/dbsprout/dbsprout/commit/6294953e153bc5ecd36cb4298f40940d58b0a2da))


### Tests

* adapt S-082 erd test to S-084 data-preview partial (reconcile) [S-084] ([a2a7ec2](https://github.com/dbsprout/dbsprout/commit/a2a7ec2626c4fda82299fcf3a651e28dc8853341))
* **DBS-101:** integration tests and throughput benchmark for direct insertion [S-042-F2] ([#67](https://github.com/dbsprout/dbsprout/issues/67)) ([68df74e](https://github.com/dbsprout/dbsprout/commit/68df74e1300c3fa1785affdbe73f9f4070b65196))
* **DBS-102:** assert clamped finite scores discriminate vs null [S-094] ([bca1b6f](https://github.com/dbsprout/dbsprout/commit/bca1b6f8b0e7d1e81f6e0f8b321215d3ebe3a53d))
* **DBS-108:** AC-1 unicode table names render verbatim [S-054b] ([a7aea30](https://github.com/dbsprout/dbsprout/commit/a7aea3033e60d8a7d029c7803230dce274c64515))
* **DBS-108:** AC-2 100-table drift scale guard [S-054b] ([a5099ea](https://github.com/dbsprout/dbsprout/commit/a5099ea386f4c2e3ef90661e1efb714dc25323c5))
* **DBS-108:** AC-3 enum and table same name render separately [S-054b] ([9d3d052](https://github.com/dbsprout/dbsprout/commit/9d3d052b96242371fae639002c7ae351e46778c2))
* **DBS-108:** AC-4 --snapshot hash + --file source [S-054b] ([f1902a3](https://github.com/dbsprout/dbsprout/commit/f1902a324e1e255cf00638a9a10d861a473d220d))
* **DBS-108:** AC-5 nonexistent output-dir exits 2 [S-054b] ([6a7a8c8](https://github.com/dbsprout/dbsprout/commit/6a7a8c86783a32e4b3b0767062d07a6ec4cbc976))
* **DBS-108:** AC-6 ~9MB DDL under cap parses [S-054b] ([8d57915](https://github.com/dbsprout/dbsprout/commit/8d579159e0c2fc49733d8782413998de8193a97d))
* **DBS-108:** dbsprout diff edge-case test coverage [S-054b] ([67e4de4](https://github.com/dbsprout/dbsprout/commit/67e4de439a940e8594b66f59a56dc8c4212f43bb))
* **DBS-108:** scaffold TestDiffEdgeCases class [S-054b] ([cc40d6b](https://github.com/dbsprout/dbsprout/commit/cc40d6b2e0bed88dffd20ae962ba87f00d541ddc))
* **DBS-113:** cover seed/freeze defensive guards in MLX trainer [S-065b] ([802744d](https://github.com/dbsprout/dbsprout/commit/802744d97df6a2d64066d85c74ba7d241f073edf))
* **DBS-113:** pin full real mlx-lm surface in integration test [S-065b] ([d382c3e](https://github.com/dbsprout/dbsprout/commit/d382c3e09ab4661c186d50ff95ea63f3697c438f))
* **DBS-120:** make test_cli_generate.py hermetic + repo-root guard [S-100] ([4b7df1e](https://github.com/dbsprout/dbsprout/commit/4b7df1e71d0baad84f2b21ef6648a3889df4fbe0))
* **DBS-120:** make test_cli_generate.py hermetic + repo-root guard [S-100] ([fafca5d](https://github.com/dbsprout/dbsprout/commit/fafca5df70278c3489ca68cac5076b607054a445))
* **DBS-73:** 100% models coverage; mypy httpx override [S-069] ([eeee852](https://github.com/dbsprout/dbsprout/commit/eeee85297a978ff073f1372b3b6efb1a9e6a991e))
* **DBS-77:** doctor edge-case coverage + bandit-safe glyph helper [S-073] ([b8f843f](https://github.com/dbsprout/dbsprout/commit/b8f843f726de42ca39551b1683b6db1d7ba81bc7))
* **DBS-78:** benchmark package + shared fixtures [S-074] ([2f04bf8](https://github.com/dbsprout/dbsprout/commit/2f04bf8f501300779a817fdbb8dcaf6be4b8ad55))
* **DBS-78:** CLI startup benchmark + python -m entry point [S-074] ([b260dde](https://github.com/dbsprout/dbsprout/commit/b260ddef7b594f11bc3ddeb094f8115fc1a23618))
* **DBS-78:** example plugin discovered via real registry [S-078] ([61e47d1](https://github.com/dbsprout/dbsprout/commit/61e47d13dca29485a65be2d8f9bdfc74f14e545f))
* **DBS-78:** generation performance benchmarks [S-074] ([f04c577](https://github.com/dbsprout/dbsprout/commit/f04c5772e379a2f2098caa4f54c72e7356575c5c))
* **DBS-78:** output + memory benchmarks [S-074] ([c35daf7](https://github.com/dbsprout/dbsprout/commit/c35daf7465e932b7e9ef09d952979c50b0c08c5d))
* **DBS-79:** E2E suite — 5 realistic schemas init→generate→validate [S-075] ([44f5454](https://github.com/dbsprout/dbsprout/commit/44f545409a109884de8afe6c9266706cd0f22d32))
* **DBS-80:** cover run() entrypoint; full quality gates pass [S-076] ([a92498f](https://github.com/dbsprout/dbsprout/commit/a92498f31394d764596dbeeaea65a471e309b285))
* **DBS-81:** failing strict mkdocs build gate [S-077] ([c890bdc](https://github.com/dbsprout/dbsprout/commit/c890bdce8821bcbefc403527cc7182523cbc0238))
* **DBS-S097:** update pipeline DP-SGD guard test for relaxed contract [S-097] ([8c93916](https://github.com/dbsprout/dbsprout/commit/8c939165d0cc4c592c84c2ee28e06e37a57868c4))
* **DBS-XX:** add E2E pipeline harness [S-075] ([681d363](https://github.com/dbsprout/dbsprout/commit/681d363cc53734e0f4d6957df67b689bc6c0a26d))
* **DBS-XX:** E2E CMS schema [S-075] ([c859e92](https://github.com/dbsprout/dbsprout/commit/c859e92593c1027f4b8c038d5c653ab990214f0b))
* **DBS-XX:** E2E e-commerce schema [S-075] ([6dbb5d4](https://github.com/dbsprout/dbsprout/commit/6dbb5d4434538c7f76140d2b3535a13659b6c0a8))
* **DBS-XX:** E2E financial schema [S-075] ([2e03a35](https://github.com/dbsprout/dbsprout/commit/2e03a35337c663fd8f74a85430af58205a5d8b61))
* **DBS-XX:** E2E SaaS multi-tenant schema [S-075] ([698a141](https://github.com/dbsprout/dbsprout/commit/698a14105694b8224280e645b6fc66978768bd20))
* **DBS-XX:** E2E social media schema [S-075] ([145b6d1](https://github.com/dbsprout/dbsprout/commit/145b6d1fbfcf38703b0fd05586ae172528c24e89))
* **DBS-XX:** finalize E2E suite, drop harness self-test [S-075] ([2936e23](https://github.com/dbsprout/dbsprout/commit/2936e23095aafa13152a75fb7c269fa54248902e))
* **diff:** de-flake large-DDL timing guard under parallel load [S-101] ([d80e199](https://github.com/dbsprout/dbsprout/commit/d80e1990adb97172170fcf4524472fa5344295fa))
* **diff:** de-flake large-DDL timing guard under parallel load [S-101] ([0eb98f1](https://github.com/dbsprout/dbsprout/commit/0eb98f13f30cb9448fff267d2aaec0ba8a966be7))
* **report:** scope self-contained checks to allow Plotly CDN [S-083] ([9a4697a](https://github.com/dbsprout/dbsprout/commit/9a4697aa30a3491bbbe73efac1e3f14effac01ab))

## [0.1.6](https://github.com/dbsprout/dbsprout/compare/v0.1.5...v0.1.6) (2026-04-07)


### Features

* **DBS-46:** PostgreSQL COPY insertion via psycopg3 [S-042] ([#54](https://github.com/dbsprout/dbsprout/issues/54)) ([92b72c1](https://github.com/dbsprout/dbsprout/commit/92b72c14cc7ca2bb73ae3d105b4233293cbbd785))
* **DBS-47:** MySQL LOAD DATA insertion via pymysql [S-043] ([#55](https://github.com/dbsprout/dbsprout/issues/55)) ([aa9a05d](https://github.com/dbsprout/dbsprout/commit/aa9a05dc2224c0ddce8e2f3b05be64eb2faa9242))
* **DBS-48:** Parquet output via Polars [S-044] ([#56](https://github.com/dbsprout/dbsprout/issues/56)) ([b96c8a0](https://github.com/dbsprout/dbsprout/commit/b96c8a0ba6a3833cc5ee762637fbf7145f233d45))
* **DBS-49:** UPSERT mode for SQL output — 4 dialects [S-045] ([#57](https://github.com/dbsprout/dbsprout/issues/57)) ([568eb1b](https://github.com/dbsprout/dbsprout/commit/568eb1bf8a89de61304df80b15ee34845075e60c))
* **DBS-50:** fidelity metrics — KS, TV, correlation, cardinality [S-046] ([#58](https://github.com/dbsprout/dbsprout/issues/58)) ([542f4b6](https://github.com/dbsprout/dbsprout/commit/542f4b6464ad7b20b924926f02141b464aaf0d52))
* **DBS-51:** C2ST detection metrics for synthetic data quality [S-047] ([#59](https://github.com/dbsprout/dbsprout/issues/59)) ([8ed8b73](https://github.com/dbsprout/dbsprout/commit/8ed8b73b6a0bc82b0856f41422921b466bc18c0d))
* **DBS-52:** QualityReport Pydantic model + JSON export enhancements [S-048] ([#60](https://github.com/dbsprout/dbsprout/issues/60)) ([cdbeec6](https://github.com/dbsprout/dbsprout/commit/cdbeec615664c2b5c471562d405107b653271bb9))
* **DBS-53:** Django model introspection via _meta API [S-049] ([#61](https://github.com/dbsprout/dbsprout/issues/61)) ([e055047](https://github.com/dbsprout/dbsprout/commit/e055047e413d48d9eaa5beba954b2710d7eed3f0))
* **DBS-54:** direct insertion auto-select with fallback chain [S-050] ([#62](https://github.com/dbsprout/dbsprout/issues/62)) ([324c2a8](https://github.com/dbsprout/dbsprout/commit/324c2a81c29aab6bd623de2c46f30df4107e53a0))

## [0.1.5](https://github.com/dbsprout/dbsprout/compare/v0.1.4...v0.1.5) (2026-04-04)


### Features

* cloud LLM provider via LiteLLM + Instructor (S-036) ([#45](https://github.com/dbsprout/dbsprout/issues/45)) ([7c7d468](https://github.com/dbsprout/dbsprout/commit/7c7d468378fe67d359c0127cf0ed9e884d6e8a8e))
* Mermaid ERD parser with regex-based extraction (S-033) ([#42](https://github.com/dbsprout/dbsprout/issues/42)) ([ff1c494](https://github.com/dbsprout/dbsprout/commit/ff1c49423a007d495cd7c5b6edb9d829bb2766b9))
* Ollama provider for local LLM spec generation (S-037) ([#46](https://github.com/dbsprout/dbsprout/issues/46)) ([e5ad58f](https://github.com/dbsprout/dbsprout/commit/e5ad58f2834f5bea0a2e8b718f60806dfeb77343))

## [0.1.4](https://github.com/dbsprout/dbsprout/compare/v0.1.3...v0.1.4) (2026-03-31)


### Bug Fixes

* clean up app.py lint errors (line length, unused noqa) ([e85b08a](https://github.com/dbsprout/dbsprout/commit/e85b08adb5b714110416946a34ac0271a5b397f6))
* lazy-import all commands for minimal install (sqlalchemy optional) ([73423ee](https://github.com/dbsprout/dbsprout/commit/73423eead95e7bbb9032b14b30f84b9d76a2898f))
* lazy-import sqlalchemy in init command for minimal install ([17c5ba0](https://github.com/dbsprout/dbsprout/commit/17c5ba0351de268de2737479e29e389f9877b45c))
* pass file as str to init_command (mypy strict) ([8814f4d](https://github.com/dbsprout/dbsprout/commit/8814f4df36b48aadac6826884cc2972a70a3a2d8))
* ruff lint errors + source_file str conversion for DDL parsing ([f16c57c](https://github.com/dbsprout/dbsprout/commit/f16c57c6ac8435849c50f98750d3f606b4ff670f))


### CI/CD

* add workflow_dispatch to publish (GITHUB_TOKEN doesn't trigger cross-workflows) ([d71adae](https://github.com/dbsprout/dbsprout/commit/d71adaedc505c181b7ab0d958371a866bbf95d3d))

## [0.1.3](https://github.com/dbsprout/dbsprout/compare/v0.1.2...v0.1.3) (2026-03-31)


### Bug Fixes

* version test no longer hardcodes version string ([e0fa2cc](https://github.com/dbsprout/dbsprout/commit/e0fa2cc6ac0c2bcb6dfb4f8f0713603cd9881d55))


### Miscellaneous

* trigger release for PyPI publish ([e0fa2cc](https://github.com/dbsprout/dbsprout/commit/e0fa2cc6ac0c2bcb6dfb4f8f0713603cd9881d55))

## [0.1.2](https://github.com/dbsprout/dbsprout/compare/v0.1.1...v0.1.2) (2026-03-30)


### Features

* geo coherence with 563 US city/state/zip lookup tuples (S-029) ([#34](https://github.com/dbsprout/dbsprout/issues/34)) ([9b9af4b](https://github.com/dbsprout/dbsprout/commit/9b9af4b91d86146f0605b0900afe183dbaa96356))
* spec cache with diskcache keyed by schema_hash (S-024) ([#29](https://github.com/dbsprout/dbsprout/issues/29)) ([f5b28bb](https://github.com/dbsprout/dbsprout/commit/f5b28bb50718999fad23a5d1d9af6edd3c98d666))


### CI/CD

* PyPI publish workflow with trusted publishing + metadata fixes (S-031) ([#36](https://github.com/dbsprout/dbsprout/issues/36)) ([cee1554](https://github.com/dbsprout/dbsprout/commit/cee1554b562922448ee742a02a0c7661ff5a27d2))

## [0.1.1](https://github.com/dbsprout/dbsprout/compare/v0.1.0...v0.1.1) (2026-03-30)


### Features

* deterministic per-column seeding via SHA-256 (S-015) ([#16](https://github.com/dbsprout/dbsprout/issues/16)) ([ef4d013](https://github.com/dbsprout/dbsprout/commit/ef4d013bc7cd1218419d7165746937f4071af192))


### Documentation

* add README with installation, quick start, and examples ([#11](https://github.com/dbsprout/dbsprout/issues/11)) ([b731db7](https://github.com/dbsprout/dbsprout/commit/b731db71b247bad8b5b333ea44321034b98fe40f))
* update README for v0.1.0 with generate command and full feature set ([#24](https://github.com/dbsprout/dbsprout/issues/24)) ([e23b63c](https://github.com/dbsprout/dbsprout/commit/e23b63c029d61c8d099fbd36f39f82d4dc50b1b1))


### Miscellaneous

* initialize repository ([9719ec2](https://github.com/dbsprout/dbsprout/commit/9719ec24c46f9131d1ec2a31dfce8c834dec8c47))
* re-trigger release-please after permissions fix ([1c6146c](https://github.com/dbsprout/dbsprout/commit/1c6146ca504fd07305c4f7f5624d0b7b36ec4a5f))


### CI/CD

* add Release Please for automated versioning and changelog ([#25](https://github.com/dbsprout/dbsprout/issues/25)) ([fd1f14c](https://github.com/dbsprout/dbsprout/commit/fd1f14c70414744eaa05597ea0ad9498d51d818d))
