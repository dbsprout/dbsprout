# Changelog

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
