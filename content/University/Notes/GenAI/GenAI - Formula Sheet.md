---
publish: true
tags: [genai, formulas, test1]
course: Generative AI (194.207)
---

# Formula sheet (with the lecture's worked numbers)

Closed-book test: you should be able to reproduce each line and say what every symbol is. Symbols → [[GenAI - Glossary|Glossary: symbols & variables]].

## Probability (Intro)

| Name | Formula | Worked number |
|---|---|---|
| Marginal | $p(x)=\sum_y p(x,y)$ | $p(\text{moo})=14+1+1=16\%$ |
| Conditional | $p(y\mid x)=p(x,y)/p(x)$ | $p(\text{cow}\mid\text{moo})=14/16=87.5\%$ |
| Product rule | $p(x,y)=p(x)\,p(y\mid x)$ | |
| Bayes | $p(x\mid y)=p(y\mid x)p(x)/p(y)$ | |
| Chain rule (AR) | $p(x_1..x_n)=\prod_t p(x_t\mid x_{<t})$ | |

## Transformer (LM)

| Name | Formula | Note |
|---|---|---|
| Dot product | $a\cdot b=\lVert a\rVert\lVert b\rVert\cos\varphi$ | similarity ≠ relevance |
| Q, K, V | $Q=XW^Q,\ K=XW^K,\ V=XW^V$ | $n\times d_k$, $n\times d_k$, $n\times d_v$ |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k}+M)V$ | $M=-\infty$ above diagonal |
| Softmax | $e^{s_j}/\sum_i e^{s_i}$ | shift-invariant, sums to 1 |
| Output | $\text{out}_i=\sum_j\alpha_{ij}v_j$ | |
| Multi-head | $\text{Concat}(\text{head}_1..\text{head}_h)W^O$, $d_k=d_{model}/h$ | 512 / 8 = 64 |
| Permutation | $\text{Attention}(PX)=P\,\text{Attention}(X)$ | no PE, no mask |
| Sinusoidal PE | $PE(pos,2i)=\sin(pos/10000^{2i/d})$, $PE(pos,2i+1)=\cos(\cdot)$ | added to $X$ |
| RoPE angle | $\theta_i=b^{-2(i-1)/d}$; rotate pair $i$ by $m\theta_i$ | $q,k$ only |
| RoPE relative | $(R(m)q)^\top(R(n)k)=q^\top R(n-m)k$ | depends on $n-m$ |
| FFN | $\text{ReLU}(xW_1)W_2$, $d_{ff}=4d_{model}$ | per token |
| MoE | $y=\sum_{k\in\text{top-}k}p_kE_k(x)$ | router per token |
| LayerNorm | $\gamma\odot(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$ | over one token's $d$ numbers |
| Residual | $x+\text{Sub}(\text{Norm}(x))$ (pre-norm) | residual stream |

## Training (TP I)

| Name | Formula | Worked number |
|---|---|---|
| FLOPs per H100-hour | $989\cdot10^{12}\cdot3600$ | $3.56\cdot10^{18}$ (peak), ×0.35 MFU = $1.23\cdot10^{18}$ |
| Training compute | $C\approx6ND$ | $6\cdot2\text{B}\cdot4\text{T}=4.8\cdot10^{22}$ |
| Chinchilla | $D\approx20N\Rightarrow C\approx120N^2,\ N\approx\sqrt{C/120}$ | $C=4.9\cdot10^{22}\Rightarrow N\approx20$B, $D\approx0.4$T |
| Vocabulary weights | $2\cdot V\cdot d$ | $2\cdot150\text{k}\cdot2048\approx0.6$B = 30% of 2B |
| Loss | $\mathcal{L}=\text{mean}_t(-\log p(\text{true}_t))$ | $p=0.033\Rightarrow3.41$ |
| Update | $\theta\leftarrow\theta-\eta_t\Delta_t$ | |
| Clipping | if $\lVert g\rVert>1$: $g\leftarrow g/\lVert g\rVert$ | |

## Long context & post-training (TP II)

| Name | Formula | Worked number |
|---|---|---|
| KV cache / token | $2\cdot L\cdot h_{kv}\cdot d_{head}\cdot\text{bytes}$ | $2\cdot24\cdot16\cdot128\cdot2=192$ KiB |
| MLA saving | cache $d_c$ instead of $2h\,d_{head}$ | 512 vs 32,768 → 64× |
| MLA absorb | $q_tk_j^\top=(q_tW^{UK\top})c_j^\top$ | |
| DPO | $-\log\sigma\big(\beta[\text{gain}^+-\text{gain}^-]\big)$, gain $=\log p_{pol}-\log p_{ref}$ | $-\log\sigma(0.1\cdot4)=0.51$ |
| Reward model | $-\log\sigma(r^+-r^-)$ | $-\log\sigma(2.0)=0.13$ |
| REINFORCE | $-r\sum_t\log p(\text{token}_t)$ | |
| with baseline | $-(r-b)\sum_t\log p(\text{token}_t)$ | $b$ = mean 0.425 |
| KL | $\sum p_{pol}(\log p_{pol}-\log p_{ref})$ | 0.20 |
| KL-penalised reward | $r_{RM}-\beta\,\text{KL}$ | $0.8-0.1\cdot1.5=0.65$ |
| On-policy distillation | grade$_t=\log p_T-\log p_S$; $\mathcal{L}=-\sum_t\text{grade}_t\log p_S(\text{token}_t)$ | +0.15, 0, −2.71 |
| PPO per-token reward | $r_t=-\beta\,\text{KL}_t+r_{RM}[t=\text{last}]$ | |
| Return / advantage | $G_t=\sum_{t'\ge t}r_{t'}$, $A_t=G_t-V_t$ | 0.65 − 0.40 = +0.25 |
| Ratio | $\rho_t=p_{now}/p_{old}$ | 0.55/0.45 = 1.22 > 1.2 → clipped |
| PPO loss | $-\sum_t\min(\rho_tA_t,\text{clip}(\rho_t,1\pm\varepsilon)A_t)$ | ε = 0.2 |
| GRPO advantage | $A_i=(r_i-\text{mean})/\text{std}$ | +1.45, +0.29, −0.48, −1.25 |
| GRPO loss | $-\sum_{i,t}[\min(\rho A_i,\text{clip}(\rho)A_i)-\beta\text{KL}_t]$ | β = 0.04 |
| Sigmoid | $\sigma(x)=1/(1+e^{-x})$ | $\sigma(0.4)=0.60$, $\sigma(2)=0.88$ |

## Applied

| Name | Formula | Worked number |
|---|---|---|
| Elo | $E_A=1/(1+10^{(R_B-R_A)/400})$ | gap 22 → 53%, 400 → 91% |
| Memory | params × bytes/weight + KV cache | 8B @ 4 bit ≈ 4 GB |
| Decode speed | tok/s ≤ bandwidth ÷ active weight bytes | 1792 / 3.8 ≈ 470 |
| Resent history | $\sum_{i=1}^n i\approx n^2/2$ | |
| Cost | $n_{in}p_{in}+n_{cached}p_{cached}+n_{out}p_{out}$ | 0.70¢ → 0.43¢ with cache |
| pass@k | $1-(1-p)^k$ | 99.97% |
| pass^k | $p^k$ | 33% |

## Handy logs (for −log p by hand)

$\ln 2=0.69$, $\ln 3=1.10$, $\ln 10=2.30$, $\ln 0.5=-0.69$, $\ln 0.1=-2.30$, $\ln 0.8=-0.22$, $\ln 0.6=-0.51$, $\sigma(0)=0.5$, $\sigma(1)=0.73$, $\sigma(-1)=0.27$.
