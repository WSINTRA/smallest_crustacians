# Project Progress: Small ML

## Goal
Learn Machine Learning by building a text classification pipeline in Rust using the Burn framework. Starting with foundational concepts and progressing to a full training loop.

## Progress
### ✅ Completed
- **Project Setup:** Initialized Cargo project with Burn 0.21 (`wgpu` + `cpu` + `autodiff` features).
- **Hello, Tensors:** Created tensors, performed element-wise addition, multiplication, and matrix multiplication.
- **TDD in Rust:** Wrote unit tests for tensor operations. Learned `#[cfg(test)]`, `#[test]`, `assert_eq!`, and handling `Result` with `.expect()`.
- **Neural Network Basics:**
  - Defined `LinearModel` struct wrapping Burn's `Linear` layer.
  - Implemented `new()` using `LinearConfig::new(2, 1).init(device)`.
  - Implemented `forward()` pass delegating to the inner layer.
  - Created `Person` struct to represent structured input data.
- **Rust Patterns:**
  - Ownership & Borrowing (`.clone()` for tensor operations).
  - `Result` handling (`into_data().to_vec().expect()`).
  - **Trait bounds & generics:** `Tensor<B, 2>` where `B: Backend` — critical for `Module` derive.
- **Loss Function:** Implemented Mean Squared Error (MSE) from scratch with TDD.
- **Feature Normalization:** Learned why scaling features matters and applied min-max normalization.
- **Gradients & Autodiff:** Called `.backward()`, retrieved gradients, understood `.requires_grad()`.
- **Training Loop:** Built a 100-epoch loop using Burn's `SgdConfig` optimizer and `GradientsParams`.
- **Burn API Navigation:** Learned to read Burn docs (`SgdConfig`, `GradientsParams::from_grads`, `LinearConfig`).
- **Burn 0.21:** Working with `Wgpu` backend (Mac Metal support) + `Cpu` for debugging.
- **Code Organization:** Extracted `Person` struct into `src/person.rs` module.
- **Generic Module Pattern:** Fixed `from_grads` returning zero gradients by making `LinearModel` generic over `B: Backend` — the derive macro needs generics to generate correct `visit` delegation.

### ✅ Completed (Session 2 — June 8, 2026)
- **TDD — Training Loop Test:** Wrote `from_grads_loss_decreased` test that asserts loss decreases after 10 epochs. Regression guard against silent training failures.
- **`pred_classifier` Function:** Built a threshold-based classifier (predictions > 0.5 → class 1, else class 0) with TDD. Two tests: all-correct and all-positive scenarios.
- **Model Evaluation:** Added post-training evaluation block in `main()` — prints each person's prediction vs target, computes and displays accuracy (achieved 100% on 3 samples).
- **Rust Patterns:**
  - **Slices (`&[f32]`)** — function accepts a slice so it works with both arrays and Vecs.
  - **Borrowing in function calls** — `pred_classifier(&predictions, 0.5)` — passing references to avoid moving ownership.
  - **`zip` + `enumerate`** — pairing two iterators with indices for parallel iteration.
  - **`Vec::push`** — building a Vec incrementally in a loop.
- **Refactor:** Put `Person` instances into a `Vec` for indexed access during evaluation.
- **Code Quality:** Ran `cargo fmt` for standard Rust formatting.

### 🚧 In Progress
- **Test Coverage:** `test_mse_loss` was dropped during refactor and should be restored.

### 🔜 Next Steps
1. Experiment with learning rate, epoch count, and optimizer choice (SGD vs Adam)
2. Add a hidden layer (multi-layer perceptron)
3. Move toward real dataset + tokenization

## Key Concepts Learned
| Concept | Description |
|---------|-------------|
| **Tensor** | Multi-dimensional array. `Tensor<B, D>` where `B` is backend, `D` is dimensions. |
| **Weights** | Learnable parameters that scale input features. |
| **Bias** | Default offset added to predictions. Allows decision boundary to shift. |
| **Broadcasting** | Automatically stretching smaller tensors to match larger ones for element-wise math. |
| **Autodiff** | Automatic differentiation. Records operations to compute gradients via chain rule. |
| **Loss Function** | A number quantifying prediction error. MSE = average of (prediction - target)². |
| **Feature Normalization** | Scaling features to a similar range (0-1) to stabilize training. |
| **Gradient** | The "slope" of the loss with respect to each weight. Points in the direction of steepest increase. |
| **Learning Rate** | Step size for gradient descent. Too big = overshoot; too small = slow convergence. |
| **Optimizer** | Automated weight updater (e.g., SGD). Handles the math of `w = w - lr * gradient`. |
| **Epoch** | One full pass through the training data. |
| **Module Derive** | `#[derive(Module)]` generates `visit`, `map`, `into_record`, etc. **Must be generic over `B: Backend`** for correct recursion into nested modules. |
| **GradientsParams** | Burn container that bridges gradients from `backward()` to the optimizer via `from_grads()`. |
| **`visit` Delegation** | The `Module` visitor pattern walks the module tree. If `visit` doesn't recurse, `from_grads` finds zero parameters. |
| **Classification Threshold** | Raw model output is a float. Apply a threshold (0.5) to convert to binary class labels (0 or 1). |
| **Accuracy** | `correct / total` — fraction of predictions that match the true labels. |
| **Slice (`&[T]`)** | A borrowed view into a sequence. Works with both arrays and Vecs. More flexible than `Vec<T>` for function parameters. |
| **`zip` + `enumerate`** | `zip` pairs two iterators element-by-element. `enumerate` adds an index. Combined: `for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate()`. |

## Lessons Learned (Process)
- **Burn's API is strict:** Type inference often needs explicit generics (`::<MyBackend, LinearModel<MyBackend>>`).
- **Generic structs matter for derives:** `#[derive(Module)]` on a concrete struct (`Linear<MyBackend>`) fails to generate proper `visit` delegation. Making the struct generic (`LinearModel<B: Backend>`) fixes it. **Always follow the docs pattern.**
- **`Clone` conflict:** `#[derive(Module)]` already provides `Clone` — deriving it separately causes `E0119: conflicting implementations`.
- **Reading the docs is part of the work:** Burn's optimizer API doesn't match PyTorch/TF patterns. The `SgdConfig -> .init() -> .step(lr, model, grads)` pattern must be learned from `burn.dev/docs`.
- **TDD is essential for debugging ML:** We used targeted tests to isolate the gradient extraction failure from the autodiff graph itself.
- **Silent failures are the worst:** Optimizer silently skips when no gradients are found. No warnings, no panics — just flat loss.
- **Diagnostic isolation works:** We proved the bug was in the wrapper (not backend, not loss, not `from_grads` itself) by testing bare `Linear` vs wrapped `LinearModel`.
- **ML tests assert properties, not exact values:** You can't predict exact outputs (weights are random), but you can assert invariants — "loss decreases," "gradients are non-zero," "accuracy > 0."
- **Test the edges, not the middle:** A test where all predictions are correct only catches bugs in the happy path. A mixed test (some correct, some wrong) proves the accuracy math is right.
- **Separate logic from presentation:** `pred_classifier` returns classified values (logic). `main()` decides how to print them (presentation). Each has a single responsibility.
- **Borrowing errors are common and fixable:** When a function expects `&[f32]` and you pass `Vec<f32>`, just add `&`. The compiler tells you exactly what to do.

## Relevant Files
- `src/main.rs`: Core implementation (model, loss, training loop, tests).
- `src/person.rs`: Input data struct module.
- `Cargo.toml`: Dependencies (`burn = { version = "~0.21", features = ["wgpu", "cpu", "tui", "train", "autodiff"] }`).
