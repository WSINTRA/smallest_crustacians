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

### 🚧 In Progress
- **Loss Function & Training:** 
  - Need to implement a loss function (e.g., Mean Squared Error or Binary Cross Entropy).
  - Implement backward pass using Burn's autodiff.
  - Create a training loop to adjust weights via gradient descent.

### 🔜 Next Steps
1. Define a loss function to measure prediction error.
2. Implement `.backward()` to compute gradients.
3. Create an optimizer (e.g., SGD) to update weights.
4. Run a training loop over multiple epochs.
5. Evaluate model performance.

## Key Concepts Learned
| Concept | Description |
|---------|-------------|
| **Tensor** | Multi-dimensional array. `Tensor<B, D>` where `B` is backend, `D` is dimensions. |
| **Weights** | Learnable parameters that scale input features. |
| **Bias** | Default offset added to predictions. Allows decision boundary to shift. |
| **Broadcasting** | Automatically stretching smaller tensors to match larger ones for element-wise math. |
| **Autodiff** | Automatic differentiation. Records operations to compute gradients via chain rule. |

## Relevant Files
- `src/main.rs`: Core implementation (model, tests, main loop).
- `Cargo.toml`: Dependencies (`burn = { version = "0.20", features = ["cpu", "autodiff"] }`).
