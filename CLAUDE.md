# Goose — Local Working Copy

Modified block/goose for personal use. Not yet forked — changes are local only (see goose-a8e).

## Dev Environment

- **Hermit required**: `source ./bin/activate-hermit` before any cargo/just/npm commands
- **Desktop launcher**: `~/Desktop/Start Goose Dev.command` builds from source, starts server + UI
- **Split mode**: `just run-server` + `just debug-ui` — fastest iteration loop
- Rust changes need recompile + server restart. UI hot-reloads.

## Couplings

- Adding a provider requires three files: the provider module, `mod.rs` (declare), `init.rs` (register)
- `GOOSE_EXTERNAL_BACKEND=true` tells Electron to connect to a running server instead of launching its own binary
- `GOOSE_SERVER__SECRET_KEY=test` required when running server standalone

## Custom Changes

- `crates/goose/src/providers/mlx.rs` — MLX provider (phase 1). Compiles, **not tested** against real MLX server.
- `.beads/` — beads tracking initialised

## Uncommitted Changes

MLX provider and beads are uncommitted. Fork needed first (goose-a8e).
