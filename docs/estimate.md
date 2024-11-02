# GPU Memory

To estimate the GPU memory required for a model with **x** billion parameters under different scenarios, we need to consider how memory usage varies with precision (e.g., `f32`, `f16`, `int4`, `int8`) and training configurations (full training, LoRA, QLoRA, etc.). Below are the formulas for each case.

## 1. Inference Memory Usage

Assuming **x** is the number of parameters in billions, we can convert it to bytes based on precision:

### a) Inference at `f32` (32-bit floating point)

Each parameter uses 4 bytes in `f32`:
- **Memory (f32)** = **x** × 10^9 × 4 bytes = **4x** GB

### b) Inference at `f16` (16-bit floating point)

Each parameter uses 2 bytes in `f16`:
- **Memory (f16)** = **x** × 10^9 × 2 bytes = **2x** GB

### c) Inference at `int4` (4-bit integer)

Each parameter uses 0.5 bytes in `int4`:
- **Memory (int4)** = **x** × 10^9 × 0.5 bytes = **0.5x** GB

### d) Inference at `int8` (8-bit integer)

Each parameter uses 1 byte in `int8`:
- **Memory (int8)** = **x** × 10^9 × 1 byte = **x** GB

---

## 2. Training Memory Usage

Training typically requires additional memory for gradients and optimizer states. Here are the calculations:

### a) Full Training (requires activations, gradients, optimizer states)

- Model parameters: **x** × 10^9
- Optimizer states and gradients require approximately 2-3 copies of the model parameters.

Using a 3x multiplier:
- **Memory (Full Training f32)** = **4x** × 3 = **12x** GB (f32)

For `f16`, the formula is:
- **Memory (Full Training f16)** = **2x** × 3 = **6x** GB (f16)

### b) Full Training + Evaluation

Additional memory is needed to store a copy of the model for evaluation (in `f16` or `int8`):
- **Memory (Total Training + Eval)** = **Memory (Full Training)** + **Memory (Evaluation (f16 or int8))**

For example, with `f16` evaluation:
- **Memory (Full Training f32 + Eval f16)** = **12x** + **2x** = **14x** GB

### c) LoRA Training

LoRA requires storing a subset of parameters (low-rank matrices) alongside the original model. Suppose LoRA introduces an additional **0.1x** (10% of model parameters):
- **Memory (LoRA)** = **Memory (f32)** + **0.1x** GB

For training LoRA with `f16`:
- **Memory (LoRA f16)** = **2x** + **0.1x** = **2.1x** GB

### d) QLoRA Training

QLoRA uses quantized weights (like `int4` or `int8`) plus LoRA matrices, allowing efficient training at reduced precision.
- **Memory (QLoRA)** = **Memory (int4 or int8)** + **0.1x**

For `int4`:
- **Memory (QLoRA int4)** = **0.5x** + **0.1x** = **0.6x** GB

### e) LoRA Training + Evaluation

For evaluation during LoRA training, add an `f16` or `int8` copy of the model:
- **Memory (LoRA + Eval)** = **Memory (LoRA)** + **Memory (f16 or int8)**

For `f16` evaluation:
- **Memory (LoRA f16 + Eval f16)** = **2.1x** + **2x** = **4.1x** GB

### f) QLoRA Training + Evaluation

- **Memory (QLoRA + Eval)** = **Memory (QLoRA)** + **Memory (f16 or int8)**

For `int4` with `f16` evaluation:
- **Memory (QLoRA int4 + Eval f16)** = **0.6x** + **2x** = **2.6x** GB

These calculations provide rough estimates for GPU memory requirements for a model of **x** billion parameters under different conditions.

# Time Consumption

To estimate the training time for a large language model (LLM) with **x** billion parameters, use a formula that takes into account factors like the total number of training tokens, batch size, learning rate, hardware efficiency, and number of GPUs.

## Training Time Formula

The formula to estimate total training time (**T_train**) is:

**T_train** = (Total Tokens × FLOPs per Token) / (Hardware Throughput × GPU Count × Utilization Rate)

### Breakdown of Each Term

1. **Total Tokens (N_tokens)**: Number of tokens for full training, often billions or trillions.
2. **FLOPs per Token (F_token)**: Estimated as:
   - **F_token** ≈ **6** × **x** × 10^9
3. **Hardware Throughput (TFLOPs/s per GPU)**: Power of each GPU (e.g., A100 can achieve ~312 TFLOPs for `f16`).
4. **GPU Count (n_GPs)**: Number of GPUs available.
5. **Utilization Rate (Util)**: Efficiency rate (e.g., 0.8 for 80%).

### Full Formula with Units

**T_train** = (N_tokens × **6** × **x** × 10^9) / (TFLOPs/s per GPU × n_GPs × Util)

### Example Calculation

For a model with:

- Model size, **x**: 7 billion parameters
- Total tokens, **N_tokens**: 300 billion tokens
- Hardware throughput: 312 TFLOPs per GPU (`f16` on an A100)
- GPU count: 8
- Utilization rate: 0.8

Then,

1. **FLOPs per Token**:  
   - **F_token** = **6** × **7** × 10^9 = **42** × 10^9 FLOPs/token

2. **Total Training Time**:
   - **T_train** = (300 × 10^9 × 42 × 10^9) / (312 × 10^12 × 8 × 0.8) ≈ **6310 seconds** ≈ **1.75 hours**

This formula provides a rough estimate; actual time may vary due to batch size, communication efficiency, and other factors.