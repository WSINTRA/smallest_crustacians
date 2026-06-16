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

### Core Principles

1. **No math explanations** — Never use calculus, derivatives, chain rule, or mathematical derivations. The user has dyscalculia. ML operations are black boxes: describe inputs, outputs, and observable behavior.
2. **Observe → Pattern → Principle** — Run the code first. Show the output. Ask the user what they notice. Extract the principle from observation, not theory.
3. **One concept at a time** — Don't introduce multiple new ideas in one step.
4. **Run after each step** — Verify the code compiles and produces expected output.
5. **Rust is first-class** — Ownership, error handling, module design, testing, and CLI patterns are primary learning goals, equal to ML concepts.
6. **Use analogies, not equations** — "The model is like a recipe that gets adjusted" not "the gradient descends the loss landscape."
7. **Check understanding** — Ask the user if the concept makes sense before moving on.
8. **Refer to TUTOR.md** — Follow the step-by-step plan; don't skip ahead.

### What to Say Instead

| Instead of this | Say this |
|---|---|
| "The gradient tells us the direction of steepest descent" | "The gradient is a number the library computes that tells the optimizer which way to adjust each weight" |
| "Loss is the mean squared error, which is the average of squared differences" | "Loss is a single number: higher means the model is more wrong, lower means it's more right" |
| "Backpropagation applies the chain rule" | "`.backward()` walks backward through the model and figures out how much each weight contributed to the error" |
| "Learning rate controls the step size in parameter space" | "Learning rate is a dial: bigger means bigger adjustments per step, smaller means smaller adjustments" |
| "Cross-entropy measures KL divergence" | "Cross-entropy is a loss function that penalizes wrong predictions more than right ones" |

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
