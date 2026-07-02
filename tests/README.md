# Test Suite Notes

This test suite is primarily maintainer-facing. It is intentionally not part of
the user-facing documentation for running design campaigns.

Many tests in this folder were AI-generated during rapid pipeline development.
They are useful as regression coverage for fragile areas such as config parsing,
campaign resume semantics, SQLite state transitions, structure-target indexing,
validation bookkeeping, export schemas, framework assets, and loss wiring.
However, the suite is admittedly excessive in places:

- some tests assert private implementation details rather than stable behavior;
- several files contain duplicated setup and synthetic YAML fixtures;
- mocked "GPU smoke" tests are runtime contract tests, not proof of real GPU
  execution;
- long integration-style tests may be better split into focused unit,
  integration, and optional hardware smoke layers.

Do not treat this folder as an example of the public API or recommended user
workflows. Human users should start with the top-level `README.md`, the `guide/`
docs, and the example configs. Maintainers should feel free to simplify,
rename, or delete tests that only preserve accidental implementation details, as
long as the important regression contracts remain covered.
