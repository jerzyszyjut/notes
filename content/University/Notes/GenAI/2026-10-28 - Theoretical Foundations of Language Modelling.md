---
publish: true
tags: [genai, lecture, test1]
lecture: 2026-10-28
lecturer: Johannes Oster
site: https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/
course: Generative AI (194.207)
---

# 02 · Theoretical Foundations of Language Modelling (28 Oct)

> [!abstract] The lecture in one picture
> To predict the next token a model needs five things, and each has its own part of the transformer:
>
> | Need | Example | Part |
> |---|---|---|
> | What each piece means | ·cow is an animal; ·bank is money or river | **token embedding** |
> | How the pieces relate | "the cow says ___": cow shapes what says means | **(masked multi-head) attention** |
> | Where each piece stands | the cat sits on the table ≠ the table sits on the cat | **positional encoding** (RoPE) |
> | What the world knows | "the Eiffel Tower is in ___": no word says Paris | **feed-forward** (+ MoE) |
> | What comes next | a probability for every token | **linear + softmax** |
>
> Plus the glue: **Add & Norm** (residual stream + LayerNorm) and **N stacked decoder blocks**.

---

## Interactive page (5w1h)

<iframe src="https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:attention" width="100%" height="750" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!tip]- Section links
> [Neural networks](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:neural-networks) · [Embeddings](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:embeddings) · [Attention](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:attention) · [Queries & keys](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:queries-keys) · [Values](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:values) · [In matrices](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:attention-matrices) · [Scores & weights](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:attention-weights) · [Masked attention](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:masked-attention) · [Multi-head](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:multi-head) · [Position](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:position) · [RoPE](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:rope) · [Feed forward](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:feed-forward) · [Add & Norm](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:add-norm) · [Linear & softmax](https://5w1h.academy/introduction-to-generative-ai/module-01/language-modelling/#sec:head)
>
> This page has static figures only. For things to play with (softmax temperature/√dₖ, the causal mask, RoPE angles) open the **Attention**, **RoPE** and **LayerNorm** tabs of [[GenAI Lab|the interactive lab]].

---

## 0 · Neural network recap (shapes)

> [!tip] In plain words
> A neural network is one big function with many knobs (parameters). Training = turn the knobs a little after every batch so the error goes down; inference = use the function with the knobs fixed.
> The shapes are just **bookkeeping**: (B, n, d) means B texts in the batch, n tokens each, d numbers per token. A language model makes **one prediction per position**, so a sentence of 3 tokens gives 3 answers ("what comes after the?", "after the cow?", "after the cow says?").

A network is a function $\hat y=f(x;\theta)$. ① initialise $\theta$ randomly → ② train with backpropagation (forward → loss → backward → each weight steps against its gradient) → ③ inference on new inputs.

Tensors by number of axes: scalar `()`, vector `(d)`, matrix `(n, d)`, tensor `(B, n, d)` = **B**atch × **n** tokens × **d** numbers per token. In the toy example "the cow says ___" with $n=3$ tokens of width 2: $X$ is (3,2), $W_1^\top$ is (2,2), $W_2^\top$ is (2,1), output $\hat Y$ is (3,1): **one prediction per position** ("is the next word moo?" at *every* word).

> [!example]- Slide 10 (click to open)
> ![[GenAI-L2-slajd10.png|720]]

---

## 1 · Embedding

> [!tip] In plain words
> Step 1 of the model: replace each token ID by its row of the embedding table. Nothing else happens here: no looking at neighbours, no context. It is a lookup, but the table itself is **learned** like every other weight.

Token → token ID → **row** of the embedding table $E\in\mathbb{R}^{V\times d_{model}}$. Qwen3-0.6B: $V$ = 151,936 rows, $d_{model}$ = 1,024. It is a lookup; it does not look at neighbours; it is learned during training like every other weight.

> [!example]- Slide 14 (click to open)
> ![[GenAI-L2-slajd14.png|720]]

---

## 2 · Attention – the core

> [!tip] In plain words
> Attention is **how each word gathers information from the other words**. Think of a meeting where every word asks a question out loud (**query**), every word wears a name tag describing what it can offer (**key**), and every word has something to say (**value**).
> Each word compares its question with all the name tags. The better the match, the more it listens to that word's message. Its new meaning = **a weighted mix of the messages** it listened to.
> Why not just compare the embeddings directly? Because **relevant ≠ similar**: "purple" is very relevant to "cow" (it changes which cow we mean) but purple and cow are not similar words. Learned queries and keys let the model decide what is relevant.

**Goal:** compute *how much* (weights) and *how* (values) each token should shift the representation of every other token.

**Why not just the dot product of embeddings?** The dot product $a\cdot b=\lVert a\rVert\lVert b\rVert\cos\varphi$ measures *similarity*, but relevance ≠ similarity ("purple" is relevant to "cow" although they are not similar). So we first project into a learned **query–key space**.

### Step by step with shapes (one head, n tokens)

> [!tip] In plain words
> Read the table top to bottom as a recipe. Three learned matrices make three versions of every token (question, name tag, message). One matrix multiplication compares **every** question with **every** name tag at once, giving an n × n table of scores. Row i = "how much does token i want to listen to each token j". Scale, hide the future, turn each row into percentages that add up to 100 %, and mix the messages with those percentages.

| Step | Formula | Shape | Plain words |
|---|---|---|---|
| input | $X$ | $n\times d_{model}$ | one row per token (embedding, + position for some PE types) |
| queries | $Q=XW^Q$ | $n\times d_k$ | what each token is looking for |
| keys | $K=XW^K$ | $n\times d_k$ | what each token offers to be found by |
| values | $V=XW^V$ | $n\times d_v$ | *how* a token moves another one, kept separate from where it sits in meaning space (using the embedding itself as the value can also work: He & Hofmann, ICLR 2024; a learned $W^V$ is the usual choice) |
| scores | $S=QK^\top$ | $n\times n$ | $s_{ij}=q_i\cdot k_j$ = how much word $j$ shapes the meaning of word $i$ |
| scale | $S/\sqrt{d_k}$ | $n\times n$ | same order, smaller range |
| mask (decoder) | $S/\sqrt{d_k}+M$ | $n\times n$ | $M_{ij}=-\infty$ for $j>i$ (future), 0 otherwise |
| weights | $\alpha=\text{softmax}_{\text{row}}(\cdot)$ | $n\times n$ | each row: numbers in [0,1] that sum to 1 |
| output | $\alpha V$ | $n\times d_v$ | $\text{out}_i=\sum_j\alpha_{ij}v_j$: weighted mix of values |

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

> [!question] Which index is which? (the most common confusion)
> - **Row $i$** = the token being updated (the *query*, "me").
> - **Column $j$** = the token being looked at (the *key/value*, "the one I attend to").
> - Softmax runs **along a row** (over $j$), so each token's weights over all others sum to 1.
> - The mask kills columns $j>i$ in row $i$.

> [!example]- Slides 18, 20 (click to open)
> ![[GenAI-L2-slajd18.png|720]]
>
> ![[GenAI-L2-slajd20.png|720]]

### Why divide by √dₖ?

> [!tip] In plain words
> A score is a sum of dₖ products. Adding up more random numbers gives a bigger spread: with dₖ = 64 the scores swing about 8× as wide as with one number. Softmax hates big gaps: if one score is much larger than the rest, it gets **almost 100 %** and everything else almost 0. Then the model is stuck: a tiny change to the scores changes nothing, so **no gradient flows back** and learning stalls.
> Dividing by √dₖ shrinks the scores back to a comfortable range, like **turning down the volume without changing which singer is loudest**: the order stays the same.

If the entries of $q$ and $k$ are roughly independent with variance 1, then $q\cdot k$ is a sum of $d_k$ terms, so its **variance grows like $d_k$** and its standard deviation like $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ keeps scores in a range where softmax is not saturated. With $d_k=64$: divide by 8 (slide: scores ÷ 8 → −5.6 … 5.6).

Without scaling: big scores → softmax gives one token ≈ all the weight and the rest ≈ 0 → gradients through softmax ≈ 0 → learning stalls. Scaling **keeps the order** of the scores (it's a positive constant), and it is *not* what makes the row sum to 1 (softmax does that). It has nothing to do with sequence length $n$.

**Try it: Attention row: √dₖ, mask, softmax**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=attention" width="100%" height="640" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slide 22 (click to open)
> ![[GenAI-L2-slajd22.png|720]]

### Softmax

> [!tip] In plain words
> Softmax turns any list of numbers into **percentages that add up to 100 %**: make every number positive with e^x, then divide by the total. Bigger numbers get bigger shares, the order never changes, and shifting all numbers by the same amount changes nothing (only the differences matter).
> Why −∞ for the mask: e^−∞ = 0, so a masked token gets exactly 0 % and the remaining tokens share the full 100 %.

$$\text{softmax}(s)_j=\frac{e^{s_j}}{\sum_i e^{s_i}}$$

- outputs in (0,1), sum to 1, keeps the order; adding the same constant to all $s_j$ changes nothing.
- $e^{-\infty}=0$, which is why the mask uses $-\infty$ **before** softmax: masked positions get exactly 0 and the remaining weights still sum to 1. Setting weights to 0 *after* softmax would leave rows summing to < 1.

> [!example]- Slide 23 (click to open)
> ![[GenAI-L2-slajd23.png|720]]

### Masked (causal) attention

> [!tip] In plain words
> When the model learns to predict word 5, it must not peek at word 5 or later, otherwise it would learn to copy instead of to predict. The mask is a **blindfold for the future**: every position can see itself and everything before it, nothing after.
> The trick: because every position sees exactly what it would see during generation, the model can practise **all positions of a text in one go**, instead of one position at a time. That is what makes training on huge texts fast.

When learning $p(x_t\mid x_1,\dots,x_{t-1})$, each prediction may only use tokens **up to and including itself**. The mask makes position $t$ see exactly what it would see during generation, so **training computes all positions in parallel** in one pass (no loop over tokens), and the sum of all per-position losses = the chain-rule log-likelihood.

> [!example]- Slide 27 (click to open)
> ![[GenAI-L2-slajd27.png|720]]

### Multi-head attention

> [!tip] In plain words
> One attention pattern can only look at the sentence in one way. Multi-head = **several readers in parallel**, each with its own questions: one might track grammar (which noun does this verb belong to?), another who "it" refers to. Each head works on a slice of the vector; at the end their findings are put side by side and mixed by one more matrix. Same total cost as one wide head, more perspectives.

Cut $Q,K,V$ into $h$ heads of width $d_{model}/h$; each head runs attention on its slice (its own "perspective", e.g. one head for syntax, one for coreference); **concatenate** the $h$ outputs (back to $d_{model}$) and mix with $W^O$. Original transformer: $d_{model}=512$, $h=8$, $d_k=d_v=64$. Cost ≈ one big head of the same total width.

```python
d_model, d_k = 512, 64
W_Q = nn.Linear(d_model, d_k, bias=False)   # same for W_K, W_V
Q, K, V = W_Q(x), W_K(x), W_V(x)            # [n, d_k] each
scores = Q @ K.T / math.sqrt(d_k)           # [n, n]
weights = scores.softmax(dim=-1)            # each row sums to 1
out = weights @ V                           # [n, d_k]
```

> [!summary] The 7 steps (slide)
> 1 make $Q,K,V$ · 2 score each query against each key · 3 divide by $\sqrt{d_k}$ · 4 in the decoder hide the future ($-\infty$ above the diagonal) · 5 softmax rows → weights summing to 1 · 6 weighted sum of values · 7 do it in $h$ heads, concatenate, project with $W^O$.

> [!example]- Slide 28 (click to open)
> ![[GenAI-L2-slajd28.png|720]]

---

## 3 · Position – attention alone has no order

> [!tip] In plain words
> Attention only compares **pairs** of words; it never asks "which one came first?". So without extra help, "dog bites man" and "man bites dog" contain exactly the same pairs, and "dog" ends up with the same vector in both: the model reads a **bag of words**, not a sentence.
> Positional information is the fix: give each token a sense of **where** it stands, either as an absolute address (position 3) or as distances between pairs (2 apart).

Attention is built from **pairs** ($q_i\cdot k_j$); the indices $i,j$ only pair words up. So without positional information and without a mask: shuffle the input → the outputs come out shuffled the same way:

$$\text{Attention}(PX)=P\cdot\text{Attention}(X)\qquad P=\text{permutation matrix}$$

"dog bites man" and "man bites dog": **dog gets the same output vector in both**, in every layer. Stacking layers doesn't help; adding the *same* vector to every token doesn't help; token IDs say *which* token, not *where*.

> [!note] Subtle point not stressed on the slides
> With a **causal mask**, attention is no longer permutation-equivariant: token 1 can only see itself, token 5 sees five tokens, so the model *can* infer some order (and "NoPE" models without positional encodings exist). The exam question specified "without a mask" for this reason.

> [!example]- Slide 32 (click to open)
> ![[GenAI-L2-slajd32.png|720]]

### Absolute vs relative

> [!tip] In plain words
> - **Learned table**: like **numbered seats** in a cinema; each seat number has its own learned vector. Simple, but there is no seat 1,025 if the hall has 1,024.
> - **Sinusoidal**: each position gets a pattern of waves of different speeds, like the **hands of a clock** (seconds, minutes, hours) that together tell the exact time. Nothing to learn, works for any position.
> - **Relative bias**: forget seat numbers, just add a bonus or penalty to each score based on **how far apart** the two words are.
> - **RoPE**: turn the query and key vectors by an angle that grows with the position (next section).

| Method | How | + | − |
|---|---|---|---|
| **Learned absolute** (GPT-2) | table $P$ with one learned row per position (1,024 rows × 768), **added** to $X$ | model learns what positions should be | fixed max length (no row 1,024); late rows trained rarely |
| **Sinusoidal absolute** (original) | $PE(pos,2i)=\sin(pos/10000^{2i/d})$, $PE(pos,2i+1)=\cos(\cdot)$, **added** to $X$ | nothing to learn, any position; "$k$ positions later" = same fixed rotation of the vector | relative position only **implicit** |
| **Relative bias** (T5, ALiBi) | add a number $b_{j-i}$ per **distance** to the **scores**: $S+B$ | directly relative | – |
| **RoPE** (Llama, Qwen) | **rotate** $q$ and $k$ by an angle proportional to the position | relative by construction; no extra parameters | needs care to extend context (→ YaRN, TP II) |

**Sinusoidal variables:** $pos$ = position (0,1,2,…), $i$ = index of the sin/cos **pair** of dimensions, $d$ = model width. Each pair is a wave over positions; low $i$ = fast wave, high $i$ = slow wave ("slower and slower to the right").

> [!example]- Slides 35, 36 (click to open)
> ![[GenAI-L2-slajd35.png|720]]
>
> ![[GenAI-L2-slajd36.png|720]]

### RoPE in detail

> [!tip] In plain words
> Imagine every query and key as a set of **clock hands**. At position m, every hand is turned by m small steps. When a query (at m) is compared with a key (at n), only the **difference** of their turns matters, so the score depends on **how far apart** the words are, not on where they are in the text. "cow says" at positions 2–3 scores like "cow says" at 12–13.
> Each pair of numbers is a hand with its own speed: **fast hands** tell neighbours apart (but go round many times, like a second hand), **slow hands** tell far distances apart (like an hour hand). Only q and k are turned; v, the message itself, is not.

$$R(m)=\text{diag}\big(R(m\theta_1),R(m\theta_2),\dots,R(m\theta_{d/2})\big),\qquad \theta_i=b^{-2(i-1)/d}$$

| Variable | Meaning |
|---|---|
| $m$ | position of the token (query at $m$, key at $n$ – here $n$ is a *position*) |
| $i$ | which **pair** of dimensions (1 … $d/2$); dims 1–2 are pair 1 |
| $\theta_i$ | angle turned **per position** by pair $i$; pair 1: $\theta_1=b^0=1$ rad, later pairs exponentially slower |
| $b$ | base: 10,000 (orig., Llama 2) · 500,000 (Llama 3, Olmo 3) · 1,000,000 (Qwen3) |
| $d$ | dimension of the vectors being rotated (the **head** dimension) |
| $R(\phi)$ | 2×2 rotation $\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}$ |

Key identity: $(R(m)q)^\top(R(n)k)=q^\top R(n-m)\,k$. The score only depends on the **distance** $n-m$: ·says at 3 vs ·cow at 2 gives the same angle difference as positions 13 and 12.

- Only **$q$ and $k$** are rotated, **not $v$** (position should affect *how much* to attend, not *what* is carried).
- Applied **inside every attention layer**, not added once at the input.
- **Different speeds per pair**: fast pairs resolve nearby order (a full turn every few tokens), slow pairs distinguish far-apart positions without wrapping around. One speed alone fails: large steps wrap around (position 7 = position 1 after 360°), tiny steps make neighbours indistinguishable.

**Try it: RoPE dials**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=rope" width="100%" height="560" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slides 38, 39, 40 (click to open)
> ![[GenAI-L2-slajd38.png|720]]
>
> ![[GenAI-L2-slajd39.png|720]]
>
> ![[GenAI-L2-slajd40.png|720]]

---

## 4 · Feed-forward – knowledge

> [!tip] In plain words
> Attention only moves information **between** the words of the sentence. But "The Eiffel Tower is in ___" needs a fact that is in no word: Paris. The feed-forward layer is where such knowledge is stored: a small network applied to **each token on its own**, like a **lookup in the model's long-term memory** ("Eiffel Tower → Paris, France").
> It holds most of the weights of a block. **Mixture of experts**: instead of one big memory, many small specialist memories, and a router sends each token to a few of them; more knowledge, same work per token.

"The Eiffel Tower is in ___": attention can only combine what is **in** the sentence; "Paris" is in no piece. The FFN applies **the same 2-layer network to each token on its own**:

$$\text{FFN}(x)=\text{ReLU}(xW_1)W_2,\qquad W_1: d_{model}\to d_{ff},\ W_2: d_{ff}\to d_{model},\ d_{ff}=4\,d_{model}$$

512 → 2,048 → 512 in the original. It holds **most of a block's weights** (2·512·2,048 ≈ 2.1M vs 4·512² ≈ 1.05M for attention) and is seen as the main store of world knowledge. Modern models use the gated **SwiGLU** variant.

**Mixture of experts (MoE):** replace the one FFN by many small expert FFNs + a **router** that picks $k$ experts **per token**; only those run, and their outputs are mixed by the router probabilities: $y=p_2E_2(x)+p_5E_5(x)$. More parameters (knowledge), same compute per token.

> [!example]- Slides 44, 47 (click to open)
> ![[GenAI-L2-slajd44.png|720]]
>
> ![[GenAI-L2-slajd47.png|720]]

---

## 5 · Add & Norm – keeping it trainable

> [!tip] In plain words
> **Add (residual)**: each layer does not replace a token's vector, it **adds a correction** to it. Picture a document passed through 28 editors where each one writes suggestions in the margin instead of rewriting the text. The original is never lost, and the training signal can flow straight back through the stack.
> **Norm (LayerNorm)**: after each step, bring the token's numbers back to a standard scale (mean 0, spread 1), then let learned γ and β stretch and shift them. Like **auto-levelling the volume** after every effect pedal so nothing explodes or fades out in a deep stack. It works on one token's own numbers, nothing else.

**Residual connection:** $x+\text{Attention}(x)$, $x+\text{FFN}(x)$: each sublayer *adds* what it found to the token's vector, which is carried straight up: the **residual stream**. Gradients flow through the identity path; deep stacks stay trainable.

**LayerNorm**, per token vector (over its $d$ numbers, *not* across the batch or across tokens):

$$\text{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\quad \mu=\tfrac1d\sum_k x_k,\quad \sigma^2=\tfrac1d\sum_k(x_k-\mu)^2$$

$\epsilon\approx10^{-6}$ (never divide by 0), $\gamma,\beta$ learned per dimension (start 1 and 0). Purpose: every token's numbers back in the same range after every sublayer, however deep → stable training. Qwen3 uses **RMSNorm** (no mean subtraction, no $\beta$) and QK-norm. Original paper: post-norm $\text{Norm}(x+\text{Sub}(x))$; modern models (TP I recap): **pre-norm** $x+\text{Sub}(\text{Norm}(x))$.

**Try it: LayerNorm on one vector**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=layernorm" width="100%" height="560" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slides 53, 54 (click to open)
> ![[GenAI-L2-slajd53.png|720]]
>
> ![[GenAI-L2-slajd54.png|720]]

---

## 6 · Linear + softmax – what comes next

> [!tip] In plain words
> After the last block each token has a vector of 1,024 numbers, but we need **one score per vocabulary entry** (151,936). The linear layer compares the vector with every token's "fingerprint" (a matrix row) and gives each a score; softmax turns the scores into probabilities.
> During training every position is scored at once; during generation only the **last** position matters, because that is where the next token goes.

After the last block: a vector of $d$ (1,024) numbers per token, but we need **one number per vocabulary entry** (151,936), and they must be **probabilities**.

1. **Linear** $W_{out}\in\mathbb{R}^{d\times V}$ → logits (any real number, negative too). In Qwen3-0.6B it **shares** the embedding table (weight tying).
2. **Softmax** row by row → a distribution over the next token at **every** position.

Training scores **every** position (row $t$ predicts token $t+1$); generation uses **only the last row**.

> [!example]- Slide 50 (click to open)
> ![[GenAI-L2-slajd50.png|720]]

---

## The full decoder-only model

> [!tip] In plain words
> Put together: look up the words, let them talk to each other (attention), let each one consult memory (feed-forward), repeat N times, then score every possible next token.

**The whole model, bottom to top** (same picture as slide 56 below):

1. **Tokens** come in: `The ·cow ·says`.
2. **Token embedding**: each token ID becomes its row of the table (a vector of d numbers).
3. **Decoder block**, repeated **N** times (28 in Qwen3-0.6B):
   1. **masked multi-head attention**, with RoPE turning q and k → **Add & Norm**
   2. **feed-forward** (or MoE) → **Add & Norm**
4. **Linear** d → V: one score (logit) per vocabulary entry.
5. **Softmax**: the scores become p(next token).

> [!example]- Slide 56 (click to open)
> ![[GenAI-L2-slajd56.png|720]]

---

## Practice questions (8)

> [!info] Same format as the test
> 0 to 4 options can be correct: decide each option on its own, true or false. Next to each option is the section above that answers it.
> Full timed tests: [[GenAI - Mock Test A|Mock A]] · [[GenAI - Mock Test B|Mock B]] · [[GenAI - Mock Test C|Mock C]] · [[GenAI - Mock Test D|Mock D]] (their questions are not repeated here).

### P1 · attention alone: no order
<small>from the official example sheet</small>

**Attention without positional encoding and without a mask reads “dog bites man” and “man bites dog”. Each word is one token. What is true?**

- [ ] **A** The word dog gets the same output vector in both <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **B** Adding the same vector to every embedding tells the two apart <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **C** Stacking more of these attention layers tells the two apart <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **D** The token IDs tell the model the order of the words <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>

> [!success]- Answer (click to open)
> **A** · A ✓ · B ✗ · C ✗ · D ✗
>
> Attention only compares pairs of tokens. Change the order of the input and you get the same outputs in the new order, in every layer. A vector added to every token is the same everywhere, and a token ID says which token it is, not where it stands.

### P2 · softmax, and why the scores are scaled
<small>from the official example sheet</small>

**Why are the attention scores divided by √dₖ before the softmax?**

- [ ] **A** Without it, large scores give one token almost all the weight <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **B** Without it, almost no gradient flows back through the softmax <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **C** Without it, the scores grow with the length of the sequence <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **D** Without it, the weights in a row do not add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Scores grow with dₖ, the length of q and k, not with the length of the sequence. With large scores, softmax gives one token nearly everything and the rest almost 0, so there is almost no gradient. Dividing keeps the order of the scores; softmax itself makes the weights add up to 1.

### P3 · masked attention: the chain rule inside attention
<small>from the official example sheet</small>

**In masked self-attention, the scores for later tokens are set to −∞ before the softmax. What is true?**

- [ ] **A** The weights for later tokens come out as exactly 0 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **B** The weights in each row still add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **C** Setting those weights to 0 after the softmax would give the same result <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **D** Because of the mask, training has to go through the tokens one after the other <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> e^−∞ = 0, so the softmax gives later tokens no weight and shares the whole 1 among the tokens up to here. Zeroing the weights after the softmax leaves rows that add up to less than 1. The mask is what lets training run every position at once: each sees exactly what it would see while generating.

### P4 · multi-head attention

**The original transformer has d_model = 512 and h = 8 heads. What is true?**

- [ ] **A** Each head works with queries and keys of length 64 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Multi-head attention|LM · Multi-head attention]]</small>
- [ ] **B** The heads’ outputs are concatenated and multiplied by W^O <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Multi-head attention|LM · Multi-head attention]]</small>
- [ ] **C** 8 heads cost about 8× the compute of one head of width 512 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Multi-head attention|LM · Multi-head attention]]</small>
- [ ] **D** Each head sees only one eighth of the tokens <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Multi-head attention|LM · Multi-head attention]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Q, K and V are cut into h slices of d_model / h = 64; every head sees all tokens, from its own perspective. Total width stays 512, so the cost is about that of one wide head.

### P5 · feed-forward

**What is true about the feed-forward layer in a decoder block?**

- [ ] **A** In a mixture of experts, the router picks experts once per sequence <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **B** It applies the same network to each token on its own <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **C** It holds most of a block’s weights <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **D** It is where information from other positions is combined <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>

> [!success]- Answer (click to open)
> **B, C** · A ✗ · B ✓ · C ✓ · D ✗
>
> 512 → 2,048 → 512: 2 · 512 · 2,048 ≈ 2.1M weights vs 4 · 512² ≈ 1.05M for attention. It turns what attention gathered into new features (e.g. Eiffel Tower → Paris). MoE routing is per token.

### P6 · values

**Why does attention use a learned value projection, v = x W^V?**

- [ ] **A** W^V is applied to the attention weights after the softmax <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **B** Using the embedding itself as the value can also work <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **C** So that how a token shifts others can differ from where it sits in meaning space <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **D** Without W^V, the attention weights could not be computed <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>

> [!success]- Answer (click to open)
> **B, C** · A ✗ · B ✓ · C ✓ · D ✗
>
> Weights come from Q and K; values decide what is moved. The simplest shift uses x itself, and that can work (He & Hofmann, ICLR 2024), but a separate projection usually does better. V = X W^V, then weights × V.

### P7 · two meanings of bank

**“He sat on the river bank” vs “She put her money in the bank”. What is true about ·bank after the last layer of Qwen3-0.6B?**

- [ ] **A** The two vectors differ <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **B** The difference comes from attention to the other words <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **C** The embedding lookups already differ <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#1 · Embedding|LM §1 Embedding]]</small>
- [ ] **D** The feed-forward layers alone could produce the difference, without attention <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Same lookup row in both; after 28 layers cosine similarity 0.79. Only attention brings in the neighbours; the feed-forward sees each token alone.

### P8 · network recap

**The toy network predicts, for “the cow says”, at every word whether the next word is moo. What is true?**

- [ ] **A** X has shape (3, 2): 3 tokens of width 2 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#0 · Neural network recap (shapes)|LM §0 Neural network recap]]</small>
- [ ] **B** It makes one prediction per position <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#0 · Neural network recap (shapes)|LM §0 Neural network recap]]</small>
- [ ] **C** In a tensor of shape (B, n, d), B is the batch size <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#0 · Neural network recap (shapes)|LM §0 Neural network recap]]</small>
- [ ] **D** Training starts from parameters copied from a smaller model <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#0 · Neural network recap (shapes)|LM §0 Neural network recap]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Initialise randomly → train with backpropagation → inference. Output Ŷ has shape (3, 1): one prediction per position, just like a language model.

---

## Exam traps from this lecture

- Only **attention** mixes information across tokens. Embedding, FFN, LayerNorm, final linear are all **per token**.
- $\sqrt{d_k}$: because scores grow with $d_k$ (vector length), **not** with $n$; prevents saturation/vanishing gradients; keeps the order; softmax (not the scaling) makes rows sum to 1.
- Mask with $-\infty$ **before** softmax → exactly 0 weight, rows still sum to 1; enables **parallel** training.
- No PE + no mask → permutation equivariant; same output vector for "dog" in both orders.
- RoPE rotates $q,k$ only, depends on relative distance, different frequency per pair, base $b$.
- FFN ≈ knowledge, holds most parameters; MoE router picks experts **per token**.
- LayerNorm normalises **within one token's vector**.
- Generation uses only the last row of logits; training uses all rows.

Related: [[GenAI - Glossary|Glossary: symbols & variables]] · [[2026-10-21 - Introduction to Generative AI|Introduction to Generative AI]] · [[2026-11-04 - Training Paradigms I|Training Paradigms I]] · [[GenAI Lab]]
