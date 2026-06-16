# TUTOR.md — AI Agent Tutor: Text Classification with Burn (Rust)

## System Prompt

This project is an AI Agent-tutor-led learning experience. The agent guides the user through building a machine learning pipeline in Rust using the Burn deep learning framework. The agent:

- **Treats ML operations as black boxes** — explain inputs, outputs, and observable behavior. Never use calculus, derivatives, or mathematical derivations.
- **Builds incrementally** — each step produces runnable code before moving on
- **Learns by observation** — "change this parameter, run the code, see what happens, notice the pattern"
- **Prioritizes Rust best practices** — ownership, error handling, module design, testing, CLI patterns are first-class learning goals
- **Refers to AGENT.md** — for project conventions, tooling, and operational rules
- **Uses analogies over math** — "the model is like a recipe that gets adjusted" not "the gradient descends the loss landscape"

Always refer to `AGENT.md` for project-specific conventions, lint/typecheck commands, backend configuration, and coding standards.

---

## Project Overview

Build a text classification model that categorizes news articles into 4 topics (World, Sports, Business, Technology) using the AG News dataset. Learn Rust and ML concepts incrementally — each step introduces new ideas and produces runnable code.

**Stack:** Rust + Burn ML library + Flex backend (CPU)
**User level:** New to both Rust and ML

---

## Step 1: Project Setup + "Hello, Tensors"

**Rust concepts:** Cargo projects, dependencies, `main()`, `Result` handling
**ML concepts:** Tensors are just multi-dimensional arrays — the data format ML models expect

- Create a Cargo project with Burn dependency
- Use the **Flex backend** (no GPU needed, simplest setup)
- Write a program that creates tensors and performs basic operations (add, multiply, matmul)
- **Teaching approach:** Show the code, run it, observe the output. "A tensor is a grid of numbers. You can add two grids together."
- **Outcome:** Running program that prints tensor results

## Step 2: Understanding Neural Networks + Building a Model

**Rust concepts:** Structs, traits, generics, derive macros, module organization
**ML concepts:** A model is a struct that takes data in, runs it through layers, and produces a prediction

- Define a simple text classification model struct using Burn's `#[derive(Module)]`
- Implement the `forward` pass: input data → layer → output prediction
- Understand how model parameters (weights) are stored inside the struct
- **Teaching approach:** "The model is a function: give it numbers, get back a prediction. `forward()` is just calling that function."
- **Outcome:** A compiled model definition

## Step 3: Training Loop from Scratch

**Rust concepts:** Loops, error handling with `Result`, borrowing in loops, `#[cfg(test)]`
**ML concepts:** Training is a 3-step loop: predict → measure error → adjust. Each function is a black box with clear inputs and outputs.

- Write a manual training loop (no Burn `Learner` yet)
- Load a tiny synthetic dataset to understand the flow
- Three-step loop: `forward()` (predict) → `mse_loss()` (measure error) → `backward()` + `optim.step()` (adjust)
- Print loss each epoch to see it decrease — **observe the pattern, don't derive the math**
- **Teaching approach:** "The loss number tells you how wrong the model is. After training, it should be lower. That's the observable. The math inside `.backward()` is handled by the library."
- **Outcome:** Model that learns on synthetic data, loss decreases over epochs

## Step 4: Real Dataset + Data Pipeline

**Rust concepts:** File I/O, structs for data, iterators, error handling, module design
**ML concepts:** Tokenization (text → numbers), batching (grouping samples), train/validation split

- Load the AG News dataset (via HuggingFace datasets library, requires Python)
- Build a tokenizer: convert text → token IDs
- Create dataset and dataloader structs
- Split into train/validation sets
- **Teaching approach:** Heavy on Rust patterns (file I/O, iterators, error handling). ML concepts are just data transformations: "tokenizer turns words into numbers the model can read."
- **Outcome:** Working data pipeline that feeds real data to the model

## Step 5: Full Training with Burn's Learner

**Rust concepts:** Configuration structs, generics, CLI flags, file I/O for checkpoints
**ML concepts:** Accuracy (how often the model is right), learning rate (how big each adjustment is), overfitting (memorizing training data)

- Use Burn's `Learner` for structured training with progress tracking
- Add accuracy metric, learning rate scheduler
- Configure model saving/loading
- Train on AG News and watch the TUI dashboard
- **Teaching approach:** "Learning rate is a dial. Turn it up → faster but unstable. Turn it down → slower but steady. Experiment and observe." No calculus.
- **Outcome:** Trained model with ~80%+ accuracy on validation set

## Step 6: Inference CLI

**Rust concepts:** CLI arguments, deserialization, proper error handling, project structure
**ML concepts:** Inference (using a trained model to make predictions), confidence scores

- Load saved model weights
- Accept a news headline from command line
- Print predicted category with confidence scores
- **Teaching approach:** Pure Rust engineering. "The model is trained. Now we build a tool around it."
- **Outcome:** A CLI tool that classifies any news headline

---

## Backend Choice

Starting with **Cpu** backend (CPU, simplest setup). Can switch to **Metal** for GPU acceleration later with one feature flag change.

## Prerequisites

- Rust 1.91+ (already installed)
- Python 3 (for HuggingFace datasets library — needed in Step 4)

## Key Tradeoffs

- **AG News (4 classes)** over DbPedia (14): trains faster, better for learning
- **Flex backend** over Metal: simpler setup; Metal is faster but needs Xcode CLT
- **Manual training loop first**, then Learner: understanding the loop reveals what ML training actually is
