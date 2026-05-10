# Project Progress: Small ML

## Goal
Learn Machine Learning by building a text classification pipeline in Rust using the Burn framework. Starting with foundational concepts and progressing to a full training loop.

## Progress
### ✅ Completed
- **Project Setup:** Initialized Cargo project with Burn 0.20 (`cpu` + `autodiff` features).
- **Hello, Tensors:** Created tensors, performed element-wise addition, multiplication, and matrix multiplication.
- **TDD in Rust:** Wrote unit tests for tensor operations. Learned `#[cfg(test)]`, `#[test]`, `assert_eq!`, and handling `Result` with `.expect()`.
- **Neural Network Basics:** 
  - Defined `LinearModel` struct with `weights` and `bias`.
  - Implemented `new()` to initialize random weights and zero bias.
  - Implemented `forward()` pass using matrix multiplication and broadcasting.
  - Created `Person` struct to represent structured input data.
- **Rust Patterns:** 
  - Ownership & Borrowing (`.clone()` for tensor operations).
  - `Result` handling (`into_data().to_vec().expect()`).
  - Trait bounds & generics (`Tensor<MyBackend, 2>`, `Device<MyBackend>`).
- **Loss Function:** Implemented Mean Squared Error (MSE) from scratch with TDD.
- **Feature Normalization:** Learned why scaling features matters and applied min-max normalization.
- **Gradients & Autodiff:** Called `.backward()`, retrieved gradients, understood `.requires_grad()`.
- **Training Loop:** Built a 100-epoch loop using Burn's `SgdConfig` optimizer and `GradientsParams`.
- **Burn API Navigation:** Learned to read Burn docs (`SgdConfig`, `GradientsParams::from_module`, `LinearConfig`).
- **Burn 0.21 Upgrade:** Migrated from 0.20 to 0.21 with `Wgpu` backend (Mac Metal support).
- **Code Organization:** Extracted `Person` struct into `src/person.rs` module.

### 🚧 In Progress
- **Training Loop Debugging:** Loss stays constant across epochs despite `backward()` populating gradients.
  - `GradientsParams::from_grads` returns 0 gradients on both `Autodiff<Wgpu>` and `Autodiff<Cpu>`
  - `GradientsParams::from_module` also returns 0 gradients
  - Direct `.grad()` lookup confirms gradients DO exist on params after `backward()`
  - Root cause: `ParamId` mismatch between `module.visit()` and `backward()` gradient store
  - Tried `.sum()` vs `.mean()` — no difference
  - Tried `.squeeze(0)` — Burn 0.21 forbids 0-D tensors
  - **Blocked:** Gradient extraction APIs appear broken in Burn 0.21

### 🔜 Next Steps
1. Check Burn 0.21 docs/issues for known `GradientsParams` bugs or API changes
2. Try Burn's `Learner` API (handles gradient extraction internally)
3. Manual weight update fallback using `.grad()` directly
4. Consider downgrading to Burn 0.20 if 0.21 has breaking changes
5. Once training works: evaluate model on 3 Person samples
6. Move toward real dataset + tokenization

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
| **ParamId** | Internal Burn identifier for tracking parameters in the autodiff graph. |
| **GradientsParams** | Burn container that bridges gradients from `backward()` to the optimizer. |

## Lessons Learned (Process)
- **Burn's API is strict:** Type inference often needs explicit generics (`::<MyBackend, LinearModel>`). `GradientsParams` requires `from_module(&mut gradients, &model)` to extract gradients tied to registered parameters.
- **`.clone()` in forward pass breaks autodiff tracking:** Cloning tensors creates new instances that lose their original parameter IDs, causing `from_module` to find 0 gradients.
- **Reading the docs is part of the work:** Burn's optimizer API doesn't match PyTorch/TF patterns. The `SgdConfig -> .init() -> .step(lr, model, grads)` pattern must be learned from `burn.dev/docs`.
- **TDD is essential for debugging ML:** We used targeted tests to isolate the gradient extraction failure from the autodiff graph itself.
- **Silent failures are the worst:** Optimizer silently skips when no gradients are found. No warnings, no panics — just flat loss.

## Relevant Files
- `src/main.rs`: Core implementation (model, loss, training loop, tests).
- `src/person.rs`: Input data struct module.
- `Cargo.toml`: Dependencies (`burn = { version = "~0.21", features = ["wgpu", "tui", "train", "autodiff"] }`).
