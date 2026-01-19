# English → Hindi Neural Machine Translation NMT

### Encoder–Decoder and Attention Mechanisms Implemented from Scratch (PyTorch)

This repository implements an **English → Hindi Neural Machine Translation (NMT)** system using a **sequence-to-sequence Encoder–Decoder mechanism** and an **Attention mechanism**, built directly in **PyTorch** using Jupyter notebooks.

The focus of this project is not only translation output, but **how architectural design choices affect training time, loss convergence, and optimization efficiency**, supported by **measured numerical results**.

---

## 🔍 The Training-Time Paradox

Under identical data, preprocessing, learning rate, and training steps:

> **The Encoder–Decoder mechanism requires significantly more wall-clock training time than the Attention mechanism, despite having fewer parameters.**

This behavior is directly observed in the training logs and loss values recorded during execution.

---

## 📦 Dataset and Preprocessing

| Item | Description |
|----|----|
| Sentence pairs | 5,000 English–Hindi |
| Tokenization | Independent vocabularies |
| Special tokens | `<SOS>`, `<EOS>`, `<PAD>` |
| Padding | Fixed-length sequences |
| Data usage | Same dataset for both mechanisms |

All sentences are converted into indexed sequences with explicit boundaries to maintain consistent decoding behavior.

---

## 🧠 Model Architecture Overview

### Encoder
- Learns English token embeddings
- Processes sequences sequentially
- Outputs final hidden state and time-step representations

### Decoder (Encoder–Decoder Mechanism)
- Generates Hindi tokens step-by-step
- Uses **only the final encoder state** as context

### Decoder (Attention Mechanism)
- Retains **all encoder time-step outputs**
- Computes attention weights dynamically
- Builds a context vector at each decoding step

---

## ⚙️ Training Configuration

| Parameter | Value |
|----|----|
| Framework | PyTorch |
| Training steps | 320 |
| Loss type | Accumulated (total loss) |
| Execution | Jupyter notebooks |
| Dataset size | 5,000 sentence pairs |

---

## 📊 Encoder–Decoder Mechanism — Training Results

Training was performed for **two epochs**.

| Epoch | Training Time | Total Loss |
|----|----|----|
| Epoch 1 | ~2h 2m | 2364.3392 |
| Epoch 2 | ~2h 2m | 2203.7608 |

**Total training time:** **4h 4m 16s**

**Average loss per step (Epoch 2):**

2203.7608 / 320 = 6.8868


---

## ⚡ Attention Mechanism — Training Results

Training was conducted with **identical data and step count**.

| Run | Training Time | Total Loss |
|----|----|----|
| Run 1 | 1h 7m 20s | 2280.3328 |
| Run 2 | 1h 7m 19s | 2125.05 |

**Total training time:** **2h 14m 39s**

**Average loss per step (Run 2):**

2125.05 / 320 = 6.6408


📉 **Lower loss achieved in nearly half the training time**
---

## ⏳ Why the Encoder–Decoder Mechanism Trains Slower

In the Encoder–Decoder mechanism, the decoder depends entirely on a **single compressed encoder representation**. During training:

- Gradients must repeatedly traverse long temporal paths
- Distant source information must be relearned implicitly
- Optimization updates become less effective per step

This increases **wall-clock time**, even when the parameter count is lower.

---


#### 🧪 Observed Training Behavior

- Decoder directly accesses full encoder context
- Gradient flow remains effective across longer sequences
- Alignment between source and target tokens is resolved during decoding
- Fewer redundant optimization steps are required

This explains the **training-time reduction despite added parameters**.


#### 📝 Translation Characteristics

- Hindi sentence structure is consistently formed
- `<EOS>` correctly terminates generation
- Attention improves stability for longer inputs
- Errors primarily reflect dataset size and training duration

Generated outputs demonstrate **valid token alignment and grammatical flow**, even under constrained compute.

---

## 🔁 Re-training and Adaptation

The notebooks include executable cells that allow:

- Re-training with extended epochs
- Training on custom datasets
- Scaling on higher-capacity GPU systems

Improved accuracy can be achieved by **increasing training duration or compute**, without modifying architecture.

---

## 📊 Training Under Real-World Constraints

The observed training loss is not minimized to its theoretical optimum, as both models were trained on a limited computational setup without access to high-end servers or large-scale GPU resources, reflecting common real-world conditions. Training was intentionally restricted to **two epochs** to highlight architectural behavior such as convergence speed and training efficiency rather than peak accuracy. With extended training schedules (**e.g., 30–40+ epochs**), larger datasets (beyond the current **5,000 sentence pairs**), and higher-capacity GPUs, the same architectures are expected to achieve significantly lower loss and evolve into **production-level translation models**.

---

## 🧩 Architectural Implications

While attention mitigates fixed-context limitations, training remains sequential and non-parallel across time steps. This highlights the architectural boundaries of recurrent designs and motivates further evolution in sequence modeling.

