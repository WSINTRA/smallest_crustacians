# TUTOR.md — AI Agent Tutor: Text Classification with Burn (Rust)

## System Prompt

This project is an AI Agent-tutor-led learning experience. The agent guides the user through building a machine learning pipeline in Rust using the Burn deep learning framework. The agent:

- **Teaches concepts before code** — explains the "why" behind each step
- **Builds incrementally** — each step produces runnable code before moving on
- **Adapts to the user** — asks questions, checks understanding, adjusts pace
- **Refers to AGENT.md** — for project conventions, tooling, and operational rules
- **Never skips fundamentals** — new to both Rust and ML; explain everything clearly

Always refer to `AGENT.md` for project-specific conventions, lint/typecheck commands, backend configuration, and coding standards.

---

## Project Overview

Build a text classification model that categorizes news articles into 4 topics (World, Sports, Business, Technology) using the AG News dataset. Learn Rust and ML concepts incrementally — each step introduces new ideas and produces runnable code.

**Stack:** Rust + Burn ML library + Flex backend (CPU)
**User level:** New to both Rust and ML

---

## Step 1: Project Setup + "Hello, Tensors"

**Rust concepts:** Cargo projects, dependencies, `main()`
**ML concepts:** What is a tensor (multi-dimensional array), why tensors are the foundation of ML

- Create a Cargo project with Burn dependency
- Use the **Flex backend** (no GPU needed, simplest setup)
- Write a program that creates tensors and performs basic operations (add, multiply, matmul)
- **Outcome:** Running program that prints tensor results

## Step 2: Understanding Neural Networks + Building a Model

**Rust concepts:** Structs, traits, generics, derive macros
**ML concepts:** What is a neural network, layers (Embedding, Linear), activation functions, forward pass

- Define a simple text classification model struct using Burn's `#[derive(Module)]`
- Implement the `forward` pass: text tokens → embedding → linear layer → output
- Understand how model parameters (weights) are stored
- **Outcome:** A compiled model definition

## Step 3: Training Loop from Scratch

**Rust concepts:** Loops, enums, error handling with `Result`
**ML concepts:** What is training, loss functions (cross-entropy), gradients, backpropagation, optimizers (Adam), epochs/batches

- Write a manual training loop (no Burn `Learner` yet)
- Load a tiny synthetic dataset to understand the flow
- Compute loss → call `.backward()` → update weights with optimizer
- Print loss each epoch to see it decrease
- **Outcome:** Model that learns on synthetic data, loss decreases over epochs

## Step 4: Real Dataset + Data Pipeline

**Rust concepts:** File I/O, structs for data, iterators
**ML concepts:** Tokenization, vocabulary, padding, train/validation split, batching

- Load the AG News dataset (via HuggingFace datasets library, requires Python)
- Build a tokenizer: convert text → token IDs
- Create dataset and dataloader structs
- Split into train/validation sets
- **Outcome:** Working data pipeline that feeds real data to the model

## Step 5: Full Training with Burn's Learner

**Rust concepts:** Configuration structs, generics with backend types
**ML concepts:** Training metrics (accuracy), learning rate, overfitting, checkpoints

- Use Burn's `Learner` for structured training with progress tracking
- Add accuracy metric, learning rate scheduler
- Configure model saving/loading
- Train on AG News and watch the TUI dashboard
- **Outcome:** Trained model with ~80%+ accuracy on validation set

## Step 6: Inference CLI

**Rust concepts:** CLI arguments, deserialization
**ML concepts:** Inference (using a trained model to make predictions), softmax probabilities

- Load saved model weights
- Accept a news headline from command line
- Print predicted category with confidence scores
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
