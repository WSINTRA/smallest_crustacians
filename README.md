# small_ML

A step-by-step learning project for building a text classification pipeline in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

## What This Is

This project is designed for learners who are new to **both Rust and machine learning**. It walks you through building a news article classifier (4 categories: World, Sports, Business, Technology) from the ground up, introducing Rust and ML concepts incrementally.

Rather than starting with a finished example, each step produces runnable code that builds on the last — so you always understand what's happening under the hood.

## Quick Start for Learners

**All you need to start is `TUTOR.md` and `AGENT.md` in an empty directory.** That's it. Open the directory with Opencode (or your coding harness of choice) and begin with Lesson 1.

This project includes an **Opencode agent** (`.opencode/agent/ml-rust-tutor.md`) designed specifically for this course. If you're using Opencode, it will guide you through each lesson automatically.

## What You'll Build

A CLI tool that takes a news headline and predicts its category, trained on the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) dataset.

## The Learning Path

| Step | Topic | What You'll Learn |
|------|-------|-------------------|
| 1 | Hello, Tensors | Cargo projects, tensor basics, the Burn backend |
| 2 | Building a Model | Structs, traits, neural network layers, forward pass |
| 3 | Training Loop | Loss, gradients, backpropagation, optimizers |
| 4 | Real Data Pipeline | Tokenization, batching, train/validation split |
| 5 | Full Training | Burn's `Learner`, metrics, checkpoints, ~80%+ accuracy |
| 6 | Inference CLI | Loading saved models, softmax, command-line predictions |

## Stack

- **Language:** Rust (2024 edition)
- **ML Framework:** Burn 0.20
- **Backend:** Cpu (simplest setup; Metal available later)
- **Dataset:** AG News

## Prerequisites

- Rust 1.91+
- Python 3 (needed from Step 4 for HuggingFace datasets)

## Running the Project

```bash
cargo build
cargo run
```

## Project Files

- **TUTOR.md** — The step-by-step learning plan and teaching guidelines
- **AGENT.md** — Project conventions, coding standards, and operational rules

## Design Decisions

- **AG News (4 classes)** over larger datasets — trains faster, easier to follow
- **CPU backend** to start — no GPU setup friction; switchable later with one flag
- **Manual training loop before Burn's Learner** — understanding the loop first reveals what ML training actually is

## License

MIT
