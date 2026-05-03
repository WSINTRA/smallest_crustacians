# AGENT.md — Project Conventions & Operational Rules

## Project Info

- **Name:** small_ML
- **Goal:** Learn ML by building a text classification pipeline in Rust with Burn
- **Framework:** Burn ML (https://github.com/tracel-ai/burn)
- **Backend:** Flex (CPU) — feature flag `flex`
- **Dataset:** AG News (4-class news categorization)
- **User level:** New to both Rust and ML

## Coding Conventions

- Follow standard Rust style (`cargo fmt`)
- No unnecessary comments — code should be self-explanatory; explanations go in TUTOR.md or inline teaching
- Use `snake_case` for functions/variables, `PascalCase` for types
- Prefer `let` bindings with clear names over chained expressions when teaching
- Use `Result` for error handling; avoid `unwrap()` in production code, allow it in teaching examples for clarity
- Follow Burn's patterns: generic over `B: Backend`, use `#[derive(Module)]` for model structs

## Project Structure

```
small_ML/
├── AGENT.md          # This file — conventions and rules
├── TUTOR.md          # Learning plan and system prompt
├── Cargo.toml        # Project manifest
└── src/
    └── main.rs       # Main entry point (evolves through steps)
```

## Build & Run Commands

```bash
# Build
cargo build

# Run
cargo run

# Run in release mode (faster)
cargo run --release

# Format code
cargo fmt

# Check for lint issues
cargo clippy
```

## Burn-Specific Conventions

- Backend type alias: `type MyBackend = burn::backend::cpu::Cpu;`
- For training with gradients: `type MyBackend = burn::backend::autodiff::Autodiff<burn::backend::cpu::Cpu>;`
- Tensor requires dimension parameter: `Tensor<MyBackend, 2>` for 2D tensors
- Create tensors from slices via `.into()`: `let t: Tensor<MyBackend, 2> = [[1.0, 2.0], [3.0, 4.0]].into();`
- Model structs derive `Module` and `Debug`
- Forward pass takes `Tensor<B, D>` and returns `Tensor<B, D>`

## Teaching Rules (for the AI Agent)

1. **Explain before coding** — introduce the concept, then show the code
2. **One concept at a time** — don't introduce multiple new ideas in one step
3. **Run after each step** — verify the code compiles and produces expected output
4. **Check understanding** — ask the user if the concept makes sense before moving on
5. **Connect Rust and ML** — when introducing a Rust feature, explain why it matters for ML
6. **Refer to TUTOR.md** — follow the step-by-step plan; don't skip ahead

## Dependencies (Cargo.toml)

```toml
[dependencies]
burn = { version = "0.20", features = ["cpu", "autodiff"] }
```

Additional dependencies will be added as needed per step.

## Notes

- Python 3 is required for Step 4 (HuggingFace datasets)
- macOS Metal backend available later via `metal` feature flag if desired
- Recursion limit may need `#![recursion_limit = "256"]` for WGPU backend (not needed for Flex)
