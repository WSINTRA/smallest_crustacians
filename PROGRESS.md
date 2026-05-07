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

### 🚧 In Progress
- **Optimizer Integration:** Training loop runs but loss isn't decreasing yet (Burn 0.20 autodiff + optimizer API quirks with `.clone()` severing parameter IDs). Next session: resolve using Burn's built-in `Linear` layer or manual gradient application.

### 🔜 Next Steps
1. Get the training loop actually reducing loss (resolve parameter ID tracking).
2. Evaluate model performance on the 3 Person samples.
3. Move toward real dataset + tokenization.

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

## Lessons Learned (Process)
- **Burn's API is strict:** Type inference often needs explicit generics (`::<MyBackend, LinearModel>`). `GradientsParams` requires `from_module(&mut gradients, &model)` to extract gradients tied to registered parameters.
- **`.clone()` in forward pass breaks autodiff tracking:** Cloning tensors creates new instances that lose their original parameter IDs, causing `from_module` to find 0 gradients.
- **Reading the docs is part of the work:** Burn's optimizer API doesn't match PyTorch/TF patterns. The `SgdConfig -> .init() -> .step(lr, model, grads)` pattern must be learned from `burn.dev/docs`.

## Relevant Files
- `src/main.rs`: Core implementation (model, loss, training loop, tests).
- `Cargo.toml`: Dependencies (`burn = { version = "0.20", features = ["cpu", "autodiff"] }`).
