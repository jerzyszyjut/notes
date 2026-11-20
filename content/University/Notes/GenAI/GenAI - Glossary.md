---
publish: true
tags: [genai, glossary, test1]
aliases: [Symbols, Variables, Notation]
course: Generative AI (194.207)
---

# Glossary: every symbol, what it stands for

The slides and the 5w1h pages reuse letters (`d`, `t`, `T`, `n`, `L`, `ε` …) with **different meanings in different lectures**. This page pins each one down. When a formula confuses you, look it up here first.

> [!warning] Same letter, different meaning
> - **`t`**: position of a token in a sequence (autoregression) · noise level in diffusion · training step in $\eta_t$ · token index inside an answer in PPO.
> - **`T`**: number of tokens in the chain rule (Intro slide) · duration in seconds for audio · total number of diffusion steps (T = 1000).
> - **`n`**: sequence length (number of tokens in a row / context).
> - **`d`**: width of the model = $d_{model}$ (embedding dimension). In RoPE formulas it is the dimension of *one head* that is being rotated.
> - **`L`**: number of blocks (layers) in Training Paradigms · the **loss** $\mathcal{L}$ everywhere else.
> - **`ε`**: tiny constant in LayerNorm (≈10⁻⁶) · the noise vector in diffusion · the clip range in PPO/GRPO (≈0.2).
> - **`β`**: learned shift in LayerNorm · strength of the KL leash in RLHF/DPO/GRPO (0.1, 0.04).
> - **`θ`**: all model parameters (weights) · in RoPE, $\theta_i$ is the rotation angle per position for pair $i$.
> - **`h`**: number of attention heads (sometimes the hidden vector `H` after the last block).
> - **`b`**: RoPE base (10,000 / 500,000 / 1,000,000) · the **baseline** in RL ($r-b$).
> - **`p`**: a probability · in patch embeddings, $p_i$ is the **position vector** of patch $i$.
> - **`V`**: value matrix in attention · vocabulary size in "2·V·d" · value model $V_t$ in PPO.

---

## 1 · Probability & generative modelling (Intro)

| Symbol | Read as | Meaning | Example |
|---|---|---|---|
| $X, Y$ | random variables | e.g. *animal* and *sound* on the farm | |
| $x, y$ | values | one concrete data point / one answer | $x$ = a picture, $y$ = "cow" |
| $p(x)$ | "p of x" | true probability the world produces $x$ (even one that doesn't exist yet) | what a **generative** model learns |
| $p(y \mid x)$ | "p of y given x" | probability of answer $y$ given input $x$ | what a **discriminative** model learns, e.g. $p(\text{cow}\mid\text{picture})$ |
| $p(x,y)$ | joint | probability of both together | $p(\text{cow},\text{moo}) = 14\%$ |
| $p_\theta$ | model distribution | the model's approximation, with parameters $\theta$ | goal: $p_\theta(x) \approx p(x)$ |
| $\theta$ | theta | **all** weights of the network | ~0.6B numbers in Qwen3-0.6B |
| $\hat y = f(x;\theta)$ | | the network as a function; $\hat y$ = prediction | |
| $x_t$ | token at position $t$ | | "The cow says moo" → $x_1$=The … $x_4$=moo |
| $x_{<t}$ | "x before t" | shorthand for $x_1,\dots,x_{t-1}$ | |
| $T$ (Intro) or $n$ (LM) | | number of tokens in the sequence | |
| $x \in \{0,\dots,255\}^{786{,}432}$ | | a 512×512 RGB picture = 512·512·3 numbers | |
| $x \in \mathbb{R}^{f_s \cdot T}$ | | a sound: $f_s$ samples per second × $T$ seconds | 24 kHz × 2 s = 48,000 numbers |
| $f_s$ | sample rate | samples per second | 16,000 / 24,000 / 44,100 |
| $x_i \in \mathbb{R}^{192}$ | patch $i$ | one 8×8 patch × 3 colours flattened = 192 numbers | |
| $W \in \mathbb{R}^{d\times 192}$ | patch projection | **one** learned matrix, shared by all patches | |
| $W x_i \in \mathbb{R}^d$ | patch embedding | the patch as a $d$-dim vector | |
| $p_i$ | position term | vector for (row, column) of patch $i$, **added** | |
| $z_t \in \mathbb{R}^d$ | audio frame vector | encoder output for window $t$ | |
| $K$ | codebook size | number of entries to snap to (→ token ID) | 1,024 · 4,096 · 32,768 |
| $t$ (diffusion) | noise level / step | 0 = clean, $T$ = pure noise | $T = 1000$ in training, 50 steps in sampling |
| $\epsilon$ (diffusion) | noise | the random noise drawn; the network **predicts** it | |

---

## 2 · Transformer (Theoretical Foundations of Language Modelling)

| Symbol | Meaning | Shape | Original Transformer / Qwen3-0.6B |
|---|---|---|---|
| $V$ (vocab) | number of tokens in vocabulary | scalar | – / **151,936** rows (151,669 real tokens + padding) |
| $E$ | embedding table, one row per token ID | $V \times d_{model}$ | – / 151,936 × 1,024 |
| $d_{model}$ (= $d$) | width: numbers per token vector | scalar | **512** / **1,024** |
| $n$ | sequence length (tokens in the context) | scalar | |
| $X$ | the input matrix: one row per token (embedding + position) | $n \times d_{model}$ | |
| $W^Q, W^K$ | learned projections to query / key space | $d_{model} \times d_k$ | 512 × 64 |
| $W^V$ | learned projection to values | $d_{model} \times d_v$ | 512 × 64 |
| $Q = XW^Q$ | queries: "what am I looking for?" | $n \times d_k$ | |
| $K = XW^K$ | keys: "what do I offer to be found by?" | $n \times d_k$ | |
| $V = XW^V$ | values: "how I shift others if they attend to me" | $n \times d_v$ | |
| $d_k$ | length of each query/key vector (per head) | scalar | **64** |
| $d_v$ | length of each value vector (per head) | scalar | 64 (usually $=d_k$) |
| $QK^\top$ | raw attention **scores**: row $i$, column $j$ = $q_i\cdot k_j$ | $n \times n$ | |
| $s_{ij}$ | one score = how well query $i$ fits key $j$ | scalar | |
| $\sqrt{d_k}$ | scaling factor | scalar | $\sqrt{64}=8$ |
| $M$ | causal mask: $0$ on/below diagonal, $-\infty$ above | $n\times n$ | |
| $\alpha_{ij}$ | attention **weight** of token $j$ in token $i$'s update (softmax of row $i$) | scalar in [0,1], row sums to 1 | |
| $\text{out}_i=\sum_j \alpha_{ij} v_j$ | new vector of token $i$ | $d_v$ | |
| $h$ | number of heads | scalar | **8** / 16 |
| $W^O$ | output projection after concatenating heads | $(h\cdot d_v) \times d_{model}$ | 512 × 512 |
| $P$ | permutation matrix (reorders rows) | $n\times n$ | |
| $\varphi$ | angle between two vectors in $a\cdot b = \lVert a\rVert\lVert b\rVert\cos\varphi$ | | |
| $pos$ | position index (sinusoidal PE) | | 0, 1, 2 … |
| $i$ (sinusoidal / RoPE) | index of a **pair** of dimensions | 1 … $d/2$ | |
| $m, n$ (RoPE) | positions of the query token ($m$) and key token ($n$) | | ⚠ here $n$ is a *position*, not the length |
| $\theta_i$ | angle per position step for pair $i$: $\theta_i=b^{-2(i-1)/d}$ | | pair 1: $\theta_1 = 1$ rad ≈ 57°/step (fastest) |
| $b$ (RoPE base) | sets how slow the slowest pairs turn | | 10,000 (orig., Llama 2) · 500,000 (Llama 3, Olmo 3) · 1,000,000 (Qwen3) |
| $R(m\theta_i)$ | 2×2 rotation by angle $m\theta_i$ | 2×2 | |
| $B$ | relative-position bias matrix: one learned number $b_{j-i}$ per distance | $n\times n$ | T5, ALiBi |
| $d_{ff}$ | hidden width of feed-forward | scalar | **2,048** = 4·512 |
| $W_1, W_2$ | FFN up / down projections | $d_{model}\times d_{ff}$, $d_{ff}\times d_{model}$ | |
| $\mu, \sigma^2$ | mean and variance **of one token's $d$ numbers** (LayerNorm) | scalars | |
| $\gamma, \beta$ | learned scale/shift per dimension (start 1, 0) | $d$ | |
| $\odot$ | element-wise product | | |
| $N$ or $L$ | number of decoder blocks | | 6 (orig.) / **28** (Qwen3-0.6B) |
| $H$ | hidden states after the last block (+final norm) | $n\times d$ | 3 × 1,024 for "The cow says" |
| $W_{out}$ | final linear "unembedding" | $d \times V$ | 1,024 × 151,936 (tied to $E$ in Qwen3-0.6B) |
| logits | raw scores per vocabulary entry before softmax | $n\times V$ | any real number |
| $E_k(x)$, $p_k$ | expert $k$'s output, router weight for expert $k$ (MoE) | | $y = p_2E_2(x)+p_5E_5(x)$ |

---

## 3 · Training Paradigms I (compute, data, pre-training)

| Symbol | Meaning | Value in the pilot |
|---|---|---|
| $C$ | training compute in FLOPs | $4.9\cdot10^{22}$ |
| FLOP | one multiplication **or** one addition | |
| $N$ | parameters a token is multiplied with (in MoE: **active** ones; Kaplan: non-embedding) | 2B active |
| $D$ | number of training tokens | 4T |
| 6 | 2 FLOPs forward + 4 FLOPs backward, per weight per token | |
| $N_{opt}, D_{opt}$ | compute-optimal size / tokens (Chinchilla) | 20B / 0.4T for this $C$ |
| MFU | model FLOPs utilisation = useful FLOPs ÷ peak | ~35% |
| $L$ | number of blocks (depth) | 24 |
| $d$ | width (= $d_{model}$) | 2,048 |
| $d/\text{blocks}$ | aspect ratio; good range **20–250** | 2,048 / 24 ≈ 85 |
| heads × $d_{head}$ | $d = h \cdot d_{head}$ | 16 × 128 |
| $V$ | vocabulary size | ~150k |
| $2\cdot V\cdot d$ | weights for embedding + output layer | 2·150k·2,048 ≈ 0.6B ≈ 30% of 2B |
| fertility | tokens per word (per text) relative to English | Greek with GPT-2 tokenizer: 7.2× |
| rows × tokens | batch shape | 1,024 × 4,096 ≈ 4.2M tokens/step |
| `-100` | PyTorch "ignore" label: no loss at that position | image slots |
| $\mathcal{L}$ | loss = mean of $-\log p(\text{true next token})$ | starts ≈ $\ln V$ ≈ 12 |
| $\theta$ | the weights being updated | |
| $\eta_t$ | learning rate at step $t$ (from the **scheduler**) | peak e.g. 3e-4 |
| $\Delta_t$ | the step direction & size from the **optimiser** (AdamW, Muon) | |
| $\nabla$ / gradient | direction in weight space that increases the loss | clipped to norm 1.0 |

---

## 4 · Training Paradigms II (long context, post-training)

| Symbol | Meaning | Example |
|---|---|---|
| KV cache | stored $k$ and $v$ of all earlier tokens, per layer, per KV-head | |
| bytes/token of cache | $2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot \text{bytes}$ (2 = K and V) | 2·24·16·128·2 B = 192 KiB |
| $h_{kv}$ | number of K/V heads (GQA: fewer than query heads) | 8 query heads, 2 KV groups → cache ÷4 |
| $c$ / $C$ (MLA) | compressed latent per token, cached instead of K and V | $d_c = 512$ vs $h\cdot d_{head}=16{,}384$ for each of K and V |
| $W^{DKV}$ | down-projection $d_{model}\to d_c$ | 7,168 → 512 |
| $W^{UK}, W^{UV}$ | up-projections $d_c \to h\cdot d_{head}$ that rebuild K and V | 512 → 16,384 |
| YaRN scale | factor slow RoPE pairs are divided by | ÷8 for 8K → 64K |
| policy $\pi$ / $p_{policy}$ | the model being trained in post-training | |
| reference $p_{ref}$ | **frozen** copy of the SFT model | |
| $y^+, y^-$ | chosen / rejected answer | |
| gain$^\pm$ | $\log p_{policy}(y^\pm) - \log p_{ref}(y^\pm)$ (sum over tokens) | +2, −2 |
| $\beta$ (DPO) | how hard to push; also how close to stay to ref | 0.1 |
| $\sigma(x)$ | sigmoid $1/(1+e^{-x})$ | $\sigma(0)=0.5$ |
| $r$ | reward for a whole answer (reward model / checker) | 0.8 |
| $r^+, r^-$ | reward-model scores of chosen/rejected during RM training | 1.4, −0.6 |
| KL | $\sum_{\text{token}} p_{policy}\,(\log p_{policy}-\log p_{ref})$ | 0.20 |
| $\beta$ (RLHF) | KL penalty weight | 0.1 (InstructGPT-style), 0.04 (GRPO) |
| $b$ | baseline subtracted from reward | mean of group / value model |
| $r_{RM}$ | reward model score, given at **last token only** (PPO) | |
| $r_t$ | per-token reward in PPO = $-\beta\,KL_t + r_{RM}$ (RM only at last $t$) | |
| $G_t$ | return: $r_t + r_{t+1}+\dots+r_{last}$ = "what came after" | 0.65 … 0.73 |
| $V_t$ | **value model**'s prediction of $G_t$ (the baseline in PPO) | 0.40 … 0.60 |
| $A_t$ | advantage = $G_t - V_t$ ("better than expected?") | +0.25 … +0.13 |
| $\rho_t$ | ratio $p_{now}(\text{token}_t)/p_{\text{when written}}(\text{token}_t)$ | 1.00, 1.11, 1.22 |
| $\varepsilon$ | clip range (ratio kept in $[1-\varepsilon,1+\varepsilon]$) | 0.2 |
| $G$ (GRPO) | number of answers sampled per prompt | 4 in the slide (8–64 in practice) |
| $A_i$ (GRPO) | $(r_i-\text{mean}(r))/\text{std}(r)$, same for every token of answer $i$ | +1.45, +0.29, −0.48, −1.25 |
| grade$_t$ | on-policy distillation reward per token: $\log p_{teacher}-\log p_{student}$ | +0.15, 0, −2.71 |

---

## 5 · Applied Generative AI I

| Symbol / term | Meaning | Example |
|---|---|---|
| $E_A$ | expected score (win prob.) of A vs B in Elo: $1/(1+10^{(R_B-R_A)/400})$ | gap 22 → 53% |
| $R_A, R_B$ | Elo ratings | 1526 vs 1504 |
| $p$ | probability one try succeeds | 0.8 |
| $k$ | number of tries | 5 |
| pass@k | $1-(1-p)^k$: **at least one** of k succeeds | 99.97% |
| pass^k | $p^k$: **all** k succeed | 33% |
| $n_{in}, n_{out}, n_{cached}$ | tokens in / out / read from cache | 2,000 / 300 / 1,500 |
| $p_{in}, p_{out}, p_{cached}$ | price per token for each | \$2 / \$10 / \$0.2 per M |
| TTFT | time to first token (prefill) | 0.7 s for 135 tokens |
| tokens/s | decode speed ≤ memory bandwidth ÷ bytes of active weights | 1,792 GB/s ÷ 3.8 GB ≈ 470 |
| recall@k | fraction of eval questions whose right passage is in the top-k retrieved | BM25: 10/10 in top 3 |
| $\theta$ vs context | the two levers of $p(\text{next token}\mid\text{context};\theta)$ | fine-tune vs prompt/RAG |
