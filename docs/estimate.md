# GPU memory

To estimate the GPU memory required for a model with \( x \) billion parameters under different scenarios, we need to consider how memory usage varies with precision (like `f32`, `f16`, `int4`, `int8`) and training configurations (full training, LoRA, QLoRA, etc.). Here are the formulas for each case.

### 1. **Inference Memory Usage**

Assuming \( x \) is the number of parameters in billions, we can convert it to bytes based on precision:

#### a) **When inference at `f32` (32-bit floating point)**

Each parameter uses 4 bytes in `f32`:
$\text{Memory}_{\text{f32}} = x \times 10^9 \times 4 \; \text{bytes} = 4x \; \text{GB}$

#### b) **When inference at `f16` (16-bit floating point)**

Each parameter uses 2 bytes in `f16`:
$\text{Memory}_{\text{f16}} = x \times 10^9 \times 2 \; \text{bytes} = 2x \; \text{GB}$

#### c) **When inference at `int4` (4-bit integer)**

Each parameter uses 0.5 bytes in `int4`:
$\text{Memory}_{\text{int4}} = x \times 10^9 \times 0.5 \; \text{bytes} = 0.5x \; \text{GB}$

#### d) **When inference at `int8` (8-bit integer)**

Each parameter uses 1 byte in `int8`:
$\text{Memory}_{\text{int8}} = x \times 10^9 \times 1 \; \text{bytes} = x \; \text{GB}$

---

### 2. **Training Memory Usage**

Training typically requires additional memory for gradients and optimizer states. Here are typical calculations:

#### a) **Full Training (requires activations, gradients, optimizer states)**

- Model parameters: $\ x \times 10^9 \$

- Optimizer states and gradients typically require 2-3 copies of the model parameters.

Using a 3x multiplier:

$\text{Memory}_{\text{Full Training (f32)}} = 4x \times 3 = 12x \; \text{GB (f32)}$

For `f16`, the formula is:

$\text{Memory}_{\text{Full Training (f16)}} = 2x \times 3 = 6x \; \text{GB (f16)}$

#### b) **Full Training + Evaluation**

Additional memory is needed to store a copy of the model for evaluation (in `f16` or `int8`):

$\text{Memory}_{\text{Full Training + Eval}} = \text{Memory}_{\text{Full Training}} + \text{Memory}_{\text{f16 or int8}}$

For example, with `f16` evaluation:

$\text{Memory}_{\text{Full Training (f32) + Eval (f16)}} = 12x + 2x = 14x \; \text{GB}$

#### c) **LoRA Training**

LoRA requires storing a subset of parameters (low-rank matrices) alongside the original model. Suppose LoRA introduces an additional 0.1x (10% of model parameters):

$\text{Memory}_{\text{LoRA}} = \text{Memory}_{\text{f32}} + 0.1x \; \text{GB}$

For training LoRA with `f16`:

$\text{Memory}_{\text{LoRA (f16)}} = 2x + 0.1x = 2.1x \; \text{GB}$

#### d) **QLoRA Training**

QLoRA uses quantized weights (like int4 or int8) plus LoRA matrices, allowing efficient training at reduced precision.

$\text{Memory}_{\text{QLoRA}} = \text{Memory}_{\text{int4 or int8}} + 0.1x$

For `int4`:

$\text{Memory}_{\text{QLoRA (int4)}} = 0.5x + 0.1x = 0.6x \; \text{GB}$

#### e) **LoRA Training + Evaluation**

For evaluation during LoRA training, add an `f16` or `int8` copy of the model:

$\text{Memory}_{\text{LoRA + Eval}} = \text{Memory}_{\text{LoRA}} + \text{Memory}_{\text{f16 or int8}}$

For `f16` evaluation:

$\text{Memory}_{\text{LoRA (f16) + Eval (f16)}} = 2.1x + 2x = 4.1x \; \text{GB}$

#### f) **QLoRA Training + Evaluation**

$\text{Memory}_{\text{QLoRA + Eval}} = \text{Memory}_{\text{QLoRA}} + \text{Memory}_{\text{f16 or int8}}$

For `int4` with `f16` evaluation:

$\text{Memory}_{\text{QLoRA (int4) + Eval (f16)}} = 0.6x + 2x = 2.6x \; \text{GB}$

These calculations provide rough estimates for GPU memory requirements for a model of \( x \) billion parameters under different conditions.

# Time consumming

To estimate the training time for a large language model (LLM) with \( x \) billion parameters, you can use a formula that takes into account factors like the total number of training tokens, batch size, learning rate, hardware efficiency, and number of GPUs. Here's a breakdown of a general formula and its components.

---

### Training Time Formula

The formula to estimate total training time (\( T_{\text{train}} \)) can be expressed as:

$$
T_{\text{train}} = \frac{\text{Total Tokens} \times \text{FLOPs per Token}}{\text{Hardware Throughput} \times \text{GPU Count} \times \text{Utilization Rate}}
$$

### Breakdown of Each Term

1. **Total Tokens** \((N_{\text{tokens}})\)**: This is the number of tokens the model needs to process for full training. For an LLM, this is often in the billions or trillions of tokens, depending on the dataset and training configuration.

2. **FLOPs per Token** \((F_{\text{token}})\)**: The floating-point operations (FLOPs) required to process one token, often dependent on model size. A rough estimate for FLOPs per token in transformer models is:

   $$
   F_{\text{token}} \approx 6 \times x \times 10^9
   $$
   
   where \( x \) is the model size in billions of parameters. This accounts for matrix multiplications, activations, and attention computations.

3. **Hardware Throughput** \((\text{TFLOPs/s per GPU})\)**: The computational power of each GPU, measured in teraflops (TFLOPs). Modern GPUs like the A100 can achieve around 312 TFLOPs for `f16` precision, for instance. This varies by hardware and the precision used (`f32`, `f16`, `int8`, etc.).

4. **GPU Count** \((n_{\text{GPUs}})\)**: The number of GPUs available for training.

5. **Utilization Rate** \((\text{Util})\)**: The efficiency rate, which represents the portion of hardware power effectively used during training (e.g., 0.8 for 80% utilization). This accounts for memory access, communication overhead, and other inefficiencies.

---

### Full Formula with Units

Putting it all together, the formula for training time \( T_{\text{train}} \) in seconds becomes:

$$
T_{\text{train}} = \frac{N_{\text{tokens}} \times 6 \times x \times 10^9}{\text{TFLOPs/s per GPU} \times n_{\text{GPUs}} \times \text{Util}}
$$

---

### Example Calculation

Suppose you have:

- Model size, \( x \): 7 billion parameters
- Total tokens, \( N_{\text{tokens}} \): 300 billion tokens
- Hardware throughput: 312 TFLOPs per GPU (using `f16` precision on an A100)
- GPU count: 8
- Utilization rate: 0.8 (80% efficiency)

Then,

1. **FLOPs per Token**:  
   \( F_{\text{token}} = 6 \times 7 \times 10^9 = 42 \times 10^9 \; \text{FLOPs/token} \)

2. **Total Training Time**:

   $$
   T_{\text{train}} = \frac{300 \times 10^9 \times 42 \times 10^9}{312 \times 10^{12} \times 8 \times 0.8}
   $$

   Simplifying,

   $$
   T_{\text{train}} = \frac{1.26 \times 10^{22}}{1.9968 \times 10^{15}} \approx 6310 \; \text{seconds} \approx 1.75 \; \text{hours}
   $$

This formula provides a rough estimate for training time, and actual time may vary based on batch size, hardware communication efficiency, and other factors.