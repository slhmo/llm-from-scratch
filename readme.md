# Custom Autoregressive Transformer from Scratch

An educational and high-performance implementation of an autoregressive Transformer language model built entirely from first principles using PyTorch and NumPy. This repository contains the complete mathematical and structural implementation of modern LLM components—including a Byte Pair Encoding (BPE) tokenizer, vectorized causal attention heads, layer normalization, and optimization routines—without relying on high-level abstraction libraries (e.g., `torch.nn.Transformer`, `torch.nn.LayerNorm`, or `transformers`).

The architecture is configured out-of-the-box for pre-training on the **TinyStories** dataset, demonstrating how complex linguistic structures emerge from localized matrix transformations and gradient descent.

---

## Core Architectural Highlights

*   **Custom BPE Tokenizer:** A byte-level Byte Pair Encoding tokenizer featuring GPT-4 pre-tokenization regex splits. It features an optimized chunk-tracking indexing map that circumvents the classic $O(N^2)$ merge bottleneck, achieving linear-time updates.
*   **GEMM-Optimized Multi-Head Attention:** Progresses from intuitive single-head formulations to highly optimized, parallelized Multi-Head Attention using 2D General Matrix Multiplications (GEMM) for maximum hardware utilization.
*   **Manual Layer Normalization:** A standalone implementation of LayerNorm tracking learnable scale ($\gamma$) and shift ($\beta$) parameters along the feature dimension.
*   **Custom Optimization State:** Features a from-scratch implementation of the Adam optimizer, tracking historical first and second moments with analytical bias correction (Though I essentially switched to torch.optim.AdamW for performance boost, but technically you can use this optimizer).
*   **Memory-Mapped Data Ingestion:** Utilizes NumPy binary memory maps (`np.memmap`) for zero-RAM streaming of large tokenized training datasets during batch processing.

---

## Mathematical & Algorithmic Deep-Dive

### 1. Byte Pair Encoding (BPE) Tokenization
The tokenizer maps raw text into high-density integer sequences. It operates in three structural layers:

1.  **Byte-Level Initialization:** The vocabulary is initialized with 256 core byte tokens along with user-defined special tokens (e.g., `<|endoftext|>`). This avoids out-of-vocabulary (OOV) errors.
2.  **Regex Pre-Tokenization Split:** Text is partitioned into isolated chunks using a structured regex pattern(GPT-4), matching letters, numbers, and contractions:
    ```regex
    '(?i:[sdmtlre]|ll|ve)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
    ```
    This prevents merges across distinct structural boundaries (e.g., spaces merging with punctuation).
3.  **Linear-Time Merge Optimization:** Standard BPE tracks global pair frequencies across the entire text corpus, incurring an expensive $O(N^2)$ update cost at each step. This implementation utilizes an active hash map tracking `pair_to_chunks` (mapping every pair tuple to a set of chunk indices containing it). When a merge occurs, only the affected chunks are updated:

$$\text{Affected Chunks} = \mathcal{M}[\text{best\_pair}]$$

This isolates string replacements and local pair re-evaluations, bringing the time complexity close to linear with respect to vocabulary size.

Note: For first few merges, it may take quite a lot of time(Big n(affected chunks)) but it goes drastically fast after a few merges. 

---

### 2. The Causal Attention Mechanism

#### Step A: Single-Head Attention (The Primitive Concept)
To understand the relationship between tokens, the input sequence is projected into three vector spaces: Queries ($Q$), Keys ($K$), and Values ($V$). Given an input sequence matrix $X \in \mathbb{R}^{T \times d_{\text{model}}}$, where $T$ is the sequence length and $d_{\text{model}}$ is the feature dimension, the mathematical operations are defined as:

$$Q = X W_Q, \quad K = X W_K, \quad V_{\text{up}} = X W_{V_{\text{up}}}, \quad V = V_{\text{up}} W_{V_{\text{down}}}$$

Where the parameter weights are initialized with a variance scaling factor scaled by $1/\sqrt{d_{\text{model}}}$:

$$W_Q, W_K, W_{V_{\text{up}}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{query}}}, \quad W_{V_{\text{down}}} \in \mathbb{R}^{d_{\text{query}} \times d_{\text{model}}}$$

*Note: Following low-rank decomposition techniques, the Value projection is broken into an "up-projection" and a "down-projection" matrix to compress intermediate states and stabilize spatial representations.*

The attention score matrix $A$ is computed using the scaled dot-product of Queries and Keys. To enforce causality (preventing a token from looking at future tokens), an upper-triangular causal mask $M$ is applied before the Softmax operation:

$$M_{i,j} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_{\text{query}}}} + M \right) V$$

---

#### Step B: Vectorized Multi-Head Attention (Parallel Execution)
Iterating through single heads via loops introduces significant execution overhead. The production block (`MultiHeadAttention`) vectorizes this operation using a single unified 2D General Matrix Multiplication (GEMM) projection. 

Instead of keeping separate parameters for each head, the model allocates global weight projection matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

1.  **Unified Linear Projection:** The incoming input $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ is multiplied by the global projection matrices:
    $$\mathcal{Q} = X W_Q, \quad \mathcal{K} = X W_K, \quad \mathcal{V} = X W_V$$
2.  **Tensor Reshaping & Transposition:** The resulting matrices are split into $h$ heads, each with a dimensionality of $d_{\text{head}} = d_{\text{model}} / h$:
    $$\mathcal{Q} \xrightarrow{\text{reshape}} (B, T, h, d_{\text{head}}) \xrightarrow{\text{transpose}} (B, h, T, d_{\text{head}})$$
3.  **Parallel Scaled Dot-Product:** Attention computations are calculated across all batches and heads simultaneously using tensor broadcasting:
    $$\text{Scores} = \frac{\mathcal{Q} \mathcal{K}^T}{\sqrt{d_{\text{head}}}} + M$$
    $$\text{Patterns} = \text{Softmax}(\text{Scores}, \text{dim}=-1)$$
    $$\text{Output}_{\text{parallel}} = \text{Patterns} \cdot \mathcal{V} \in \mathbb{R}^{B \times h \times T \times d_{\text{head}}}$$
4.  **Re-Concatenation & Out-Projection:** The heads are re-arranged back to the structural shape $(B, T, d_{\text{model}})$ via a contiguous layout conversion and passed through an outer mixing projection matrix $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:
    $$\text{Final Output} = \left( \text{Concatenate}(\text{head}_1, \dots, \text{head}_h) \right) W_O$$

---

### 3. Custom Layer Normalization (`CustomLayerNorm`)
Layer normalization provides vertical stabilization across the residual stream, preventing gradient explosion or vanishing behaviors. Unlike Batch Normalization, LayerNorm computes statistics independently for each sequence token across its feature dimensions.

For a specific token vector $x \in \mathbb{R}^{d_{\text{model}}}$ within a batch sequence:

1.  **Mean Calculation:**
    $$\mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_i$$
2.  **Variance Calculation:**
    $$\sigma^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_i - \mu)^2$$
3.  **Standardization:**
    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
4.  **Learnable Affine Transformation:**
    $$y_i = \gamma_i \hat{x}_i + \beta_i$$

Where $\epsilon = 10^{-5}$ is a small numerical anchor avoiding division-by-zero, and $\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}$ are learnable parameters initialized to 1 and 0 respectively.

---

### 4. Feed-Forward Network (MLP Block)
Each Transformer block contains a Position-Wise Feed-Forward Network processing token representations independently. It applies an up-projection to a higher dimensional space ($4 \times d_{\text{model}}$), introduces non-linearity via the Gaussian Error Linear Unit ($\text{GELU}$), and down-projects the result back to the residual dimension:

$$\text{MLP}(x) = \text{GELU}(x W_{\text{up}}) W_{\text{down}}$$

Where $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$ and $W_{\text{down}} \in \mathbb{R}^{4d_{	ext{model}} \times d_{\text{model}}}$.

---

### 5. Custom Adam Optimizer Formulation (`CustomAdam`)
The optimization script includes a first-principles implementation of the Adam adaptive gradient descent routine. It dynamically adjusts the learning rate for each individual parameter tensor based on a running average of its past gradients.

For each learnable weight parameter $\theta$, given a gradient $g_t = \nabla_\theta \mathcal{L}_t$ at timestep $t$:

1.  **Biased First Moment Vector (Momentum Tracking):**
    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2.  **Biased Second Raw Moment Vector (Uncentered Variance Tracking):**
    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3.  **Analytical Bias Corrections:** Because $m$ and $v$ are initialized as zero matrices, they are biased toward zero, particularly during early training steps. We correct this behavior via:
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4.  **Parameter Weight Modification:**
    $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Where $\eta$ represents the base learning rate, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.

---

## Directory Structure

```text
src/
  Model/
    attention.py       # SingleHeadAttention and GEMM-optimized MultiHeadAttention
    optimizers.py      # CustomAdam implementation from scratch
    transformer.py     # CustomLayerNorm, TransformerBlock, and full Transformer stack
  Tokenizer/
    BPE_tokenizer.py   # Tokenizer core with optimized chunk merging and GPT-4 regex split
    letter_tokenizer.py# Baseline character tokenizer for development reference
    train_tokenizer.py # Corpus evaluation script for training BPE token tables
  Configs.py           # Global hyperparameter declarations and hardware targets
  test_model.py        # Real-time interactive model evaluation loop
  tokenize_dataset.py  # Binary streaming tokenizer pipeline for datasets
  train_model.py       # Pre-training script featuring cosine schedules and warmup states
.gitignore
readme.md              # Project documentation
```

---

## Training Hyperparameters

Configuration attributes can be adjusted inside `src/Configs.py`. The baseline research values are set as follows:

| Hyperparameter | Release Value | Structural Description |
| :--- |:--------------| :--- |
| `CONTEXT_WINDOW` | `256`         | Maximum token sequence context length ($T$) |
| `N_FEATURES` | `512`         | Residual stream structural dimension ($d_{\text{model}}$) |
| `N_LAYERS` | `12`          | Total stacked `TransformerBlock` layers |
| `N_ATTENTION_HEADS`| `8`           | Head count ($h$) inside parallel MultiHead attention blocks |
| `VOCAB_SIZE` | `16384`       | Total allocation space for token identity mapping |
| `BATCH_SIZE` | `32`          | Number of concurrent training items processed |
| `LEARNING_RATE` | `0.0006`      | Base optimizer step-size velocity ($\eta$) |
| `LR_DECAY` | `True`        | Activates linear warmup followed by cosine decay schedules |

---


## 🚀 Quick Start: Evaluate Pre-trained Weights Immediately

If you want to skip the data serialization and training pipelines, you can jump straight to testing the model using a pre-trained checkpoint and tokenizer table available in the repository's **[Releases](../../releases)** section.

### 1. Download the Artifacts
Navigate to the latest release and download the bundled assets:
*   `tokenizer.pkl` — The trained byte-level BPE tokenizer state matrix. Must be placed where "TOKENIZER_PATH" in Configs.py says.
*   `transformer_weights.pt` — Pre-trained weights for the 12-layer causal Transformer block. Must be placed where "MODEL_PATH" in Configs.py says.

### 2. Set Up the Directory Structure
After you place the downloaded files into their respective target directories as defined in `src/Configs.py`, you can run:

```bash
python -m src.test_model
```
And test the model. 




## Full Execution Guide


### Step 1: Training the BPE Tokenizer
Currently, this code expects "tiny-stories.csv" dataset. Obviously you're going to have to change this code if you wish to use another dataset. 
```bash
python -m src.Tokenizer.train_tokenizer
```
*This produces a `tokenizer.pkl` state file inside your configured output destination directory.*

### Step 2: Binary Dataset Serialization
Transform the heavy CSV dataset into a continuous stream of unsigned 16-bit integers (`np.uint16`), appending individual delimiter flags (`<|endoftext|>`) between text boundaries:
```bash
python -m src.tokenize_dataset
```
*This produces a compact `tokenized_train.bin` binary file, enabling fast memory-mapped data ingestion during training.*

### Step 3: Pre-training the Transformer Model
To spin up the network weights, configure tensor core accelerations via high-precision 32-bit floating-point matrix multiplications (`torch.set_float32_matmul_precision('high')`), compile the execution graphs, and start the pre-training loop, run:
```bash
python -m src.train_model
```
*During execution, the model streams token windows from disk, logs loss steps, and periodically outputs sample text completions to track linguistic progression. The trained weights are saved to `transformer_weights.pt`.*

### Step 4: Run Interactive Inference
Inorder to load the model and to test the model's generation capabilities in real-time, launch the interactive user loop:
```bash
python -m src.test_model
```
Provide custom text prompts at the prompt boundary to evaluate the model's text generation capabilities.