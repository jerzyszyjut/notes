---
publish: true
tags: [genai, lecture, test1]
lecture: 2026-10-21
lecturer: Johannes Oster
site: https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/
course: Generative AI (194.207)
---

# 01 · Introduction to Generative AI (21 Oct)

> [!abstract] The lecture in five sentences
> AI splits into **symbolic** (explicit rules) and **subsymbolic** (learned from data = machine learning). Among learned models, **discriminative** ones learn $p(y\mid x)$ (which label fits), **generative** ones learn $p(x)$ (what the data itself looks like, so they can sample new data). Generative models make data either **piece by piece** (autoregressive, chain rule, next-token prediction) or **all at once** (diffusion, iterative denoising). Everything (text, pictures, sound, video, DNA) is first turned into **numbers → tokens**: BPE for text, patches for pictures, windows for sound. The second half of the lecture starts the transformer story: **embeddings** are a lookup table, and **context** (attention layers) is what turns "bank" into river-bank or money-bank.

## Interactive page (5w1h)

<iframe src="https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/" width="100%" height="750" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!tip] Jump straight to a section (opens the interactive widgets)
> - [Symbolic AI – rule table (Fig. 4)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:symbolic)
> - [Subsymbolic AI](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:subsymbolic)
> - [Generative vs Discriminative](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:output-modalities)
> - [Joint / conditional / marginal – clickable table (Fig. 12)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:joint)
> - [Autoregression vs Diffusion – side-by-side player (Fig. 13)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:autoregression-vs-diffusion)
> - [Autoregression – next-token sampler (Fig. 14)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:autoregression)
> - [BPE trainer + live tokenizer (Fig. 15)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:bpe)
> - [Patch explorer (Fig. 16)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:patches)
> - [Sound windows + codebooks (Fig. 17)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:sound)
> - [Diffusion forward/backward (Fig. 18–19)](https://5w1h.academy/introduction-to-generative-ai/module-01/introduction/#sec:diffusion)
>
> If the frame above stays blank, the site refuses to be framed inside Obsidian: use the links, or open them with the *Surfing* community plugin.

---

## 1 · Symbolic vs subsymbolic

| | Symbolic AI | Subsymbolic AI (machine learning) |
|---|---|---|
| Knowledge comes from | people writing explicit rules over formal symbols | examples (data) |
| Example | `if horns ∧ udder ∧ hooves ∧ tail then cow` | a network trained on cow / not-cow pictures |
| Interpretability | high: you can read the rule | low: weights are not readable |
| Fixing a mistake | edit the rule | only **indirectly**, by retraining on other data |
| Novel cases | fails outside the encoded patterns (dehorned cow ≠ cow, female goat = cow) | can generalise to unseen relationships (but may be wrong) |
| Sub-fields on the map | expert systems, search & planning, knowledge representation | statistical methods, ensembles (random forest!), neural networks |

> [!note] Trap
> A **random forest** is *subsymbolic*: it learns its splits from data, even though the result looks like rules. "Neuro-symbolic" methods combine both.

**Same parts, any task.** Every ML setup = *input* (numbers) → *model* (parameters) → *output* → *objective function* (overall error) → adjust parameters so the error goes down. Classification, regression, next word and picture generation differ only in what the output and the objective are.

> [!example]+ Slides 8, 12, 11
> ![[GenAI-L1-slajd08.png|760]]
>
> ![[GenAI-L1-slajd12.png|760]]
>
> ![[GenAI-L1-slajd11.png|760]]

## 2 · Discriminative vs generative

| | Discriminative | Generative |
|---|---|---|
| Learns | $p(y\mid x)$: which output goes with an input; **where one class ends and the next begins** (decision boundary) | $p(x)$ (or $p(x,y)$): **what the data itself looks like**, its natural shape |
| Can do | classify, score | **sample new data** like the training data (and also score how likely $x$ is) |
| Needs | examples of every class (cows *and* non-cows) | examples of the data you want to make |

The goal of both: approximate the true distribution, $p_\theta(y\mid x)\approx p(y\mid x)$ or $p_\theta(x)\approx p(x)$.

> [!info] The probabilistic view of intelligence
> "Intelligence can manifest in knowing probability distributions." Regularities of the world show up as probability differences: Qwen3-0.6B-Base finds *The cow says moo* ≈ **4×** as likely as *The cow says meow*.

> [!example]+ Slides 14, 16, 19
> ![[GenAI-L1-slajd14.png|760]]
>
> ![[GenAI-L1-slajd16.png|760]]
>
> ![[GenAI-L1-slajd19.png|760]]

## 3 · Joint, conditional, marginal

The farm table from the site (Fig. 12), $p(\text{animal},\text{sound})$:

| | moo | woof | meow | nothing | **sum** $p(\text{animal})$ |
|---|---|---|---|---|---|
| cow | 14% | 1% | 1% | 24% | **40%** |
| dog | 1% | 12% | 1% | 16% | **30%** |
| cat | 1% | 1% | 10% | 18% | **30%** |
| **sum** $p(\text{sound})$ | **16%** | 14% | 12% | 58% | 100% |

| Rule | Formula | On the table |
|---|---|---|
| Marginal (sum out the other variable) | $p(x)=\sum_y p(x,y)$ | $p(\text{moo}) = 14+1+1 = 16\%$ |
| Conditional (zoom into a row/column and renormalise) | $p(y\mid x)=\dfrac{p(x,y)}{p(x)}$ | $p(\text{cow}\mid\text{moo}) = 14/16 = 87.5\%$ |
| Product rule | $p(x,y)=p(x)\,p(y\mid x)$ | $p(\text{cow},\text{moo}) = 0.40 \cdot 0.35 = 0.14$ |
| Bayes | $p(x\mid y)=\dfrac{p(y\mid x)\,p(x)}{p(y)}$ | $p(\text{cow}\mid\text{moo}) = 0.35\cdot0.40/0.16 = 0.875$ |

> [!tip] What "normalise" means
> A conditional is just "take the slice you know is true, and divide by its total so it sums to 1 again". The same idea reappears in **softmax** and in **constrained decoding** (strike out tokens, renormalise the rest).

> [!example]+ Slide 20
> ![[GenAI-L1-slajd20.png|760]]

## 4 · Autoregressive: piece by piece

Apply the product rule again and again → **chain rule**:

$$p(x_1,\dots,x_T)=\prod_{t=1}^{T} p(x_t \mid x_{<t}) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1,x_2)\cdots$$

| Symbol | Means |
|---|---|
| $x_t$ | the token at position $t$ |
| $x_{<t}$ | all tokens before position $t$ ($x_1 \dots x_{t-1}$) |
| $T$ | number of tokens in the sequence (called $n$ in the next lecture) |
| $\prod$ | multiply all factors |

*auto* (Greek "self") + *regressive* (Latin "going back"): each value is predicted from the model's **own** previous outputs. This is **exact**, not an approximation: any joint distribution can be factored this way. The model only has to learn **one** thing: the next-token distribution $p(x_t\mid x_{<t})$.

Generation = draw token → append → ask again (Fig. 14 on the site lets you pick or "Draw").

> [!example]+ Slide 27
> ![[GenAI-L1-slajd27.png|760]]

## 5 · Diffusion: all at once

| | What happens | Who does what |
|---|---|---|
| **Forward diffusion** (training data) | take a clean example, draw noise $\epsilon$, mix in an amount set by noise level $t$ → noisy example | fixed procedure, nothing learned |
| **Learning** | network sees the noisy example (and $t$) and **predicts the noise** $\epsilon$; error = predicted vs drawn noise | the network |
| **Backward diffusion** (generation) | start from pure noise; predict the noise, subtract a bit → "a little less noisy"; repeat (e.g. 50 steps) | the network, iteratively |

Noise "type" per modality: Gaussian noise for pixels/sound/video; **masks** for text (masked diffusion: tokens replaced by `[MASK]` and un-masked step by step); **probabilities** for DNA bases (categorical noise). $T=1000$ noise levels in training; Stable Diffusion XL uses 50 sampling steps.

> [!note] AR vs diffusion in one line
> AR: many cheap steps, **one token per step**, left to right, fixed once written. Diffusion: a fixed number of steps, **every piece updated every step**, can revise.

> [!example]+ Slides 29, 30
> ![[GenAI-L1-slajd29.png|760]]
>
> ![[GenAI-L1-slajd30.png|760]]

## 6 · Tokens for every modality

| Modality | Unit | How it becomes a token | Variables |
|---|---|---|---|
| Text | **BPE** subword | start with 256 **bytes**, repeatedly merge the most frequent adjacent pair → new ID (256, 257, …); then add special tokens (`<\|endoftext\|>`, `<\|im_start\|>`, `<\|im_end\|>`) | Qwen3 vocabulary: 151,669 tokens |
| Picture | **patch** (e.g. 8×8 px) | flatten to $x_i\in\mathbb{R}^{192}$ (8·8·3), multiply by **one shared learned** $W\in\mathbb{R}^{d\times192}$, add position $p_i$ (row, column); *optional*: snap to nearest entry of a codebook (4,096) → integer ID | $d$ = embedding width |
| Sound | **window** of samples | $x\in\mathbb{R}^{f_s\cdot T}$ cut into windows → learned **encoder** → $z_t\in\mathbb{R}^d$; optional codebook, often **repeated on what is still missing** (residual vector quantisation, EnCodec) | $f_s$ = sample rate, $T$ = seconds |
| Video | patches across frames | like pictures, plus time | |
| DNA | base | A/C/G/T → 0–3 directly | |

> [!tip] BPE worked example (site Fig. 15)
> Corpus ≈ "the cow in the café says moo …". Round 1 most frequent pairs: `·c` (5×), `he`, `th`, `·t`. Merges learned in order: 256 `·c`, 257 `he`, 258 `the`, 259 `oo`, 260 `ow`, 261 `·i`, 262 `·the`, 263 `·cow`, … 270 `·moo`, 272 `·café`, 273 `·in`. Then specials 274–276.
> Tokenizing "the cows in the café say wow" → `the`(258) `·cow`(263) `s`(115) `·in`(273) `·the`(262) `·café`(272) `·say`(271) `·`(32) `w`(119) `ow`(260). **Unseen words still work**: they fall back to smaller pieces, down to single bytes. (`·` = a space glued to the front of the token.)

> [!warning] Things people mix up
> - Token **ID** = an address in the vocabulary, not a meaning. 19362 (`·cow`) is not "close to" 19363.
> - Codebook snapping for images/sound is **optional**: a model can take the continuous vectors $Wx_i+p_i$ directly (that is what most vision-language models do). A codebook is needed only if the model must **output** image/sound tokens autoregressively.
> - The patch matrix $W$ is **one** matrix for all patches; the position $p_i$ is what tells patches apart by location.

> [!example]+ Slides 31, 32, 33
> ![[GenAI-L1-slajd31.png|760]]
>
> ![[GenAI-L1-slajd32.png|760]]
>
> ![[GenAI-L1-slajd33.png|760]]

## 7 · Embeddings (start of "Language modelling")

- **Embedding table**: 151,936 rows (one per token ID) × 1,024 learned numbers (Qwen3-0.6B). Looking up `·cow` = taking **row 19362**; in code `table(torch.tensor([19362]))` equals `table.weight[19362]`.
- Dimensions have **no names** in a real model (the "animal / royalty / dairy" columns are an illustration).
- Learned because "you shall know a word by the company it keeps" (Firth). Word2Vec: predict the missing word from its neighbours, keep the vectors that make it work.
- **Near = similar** (cosine similarity: cow ↔ cows 0.75, piano ≈ 0). **Directions carry relations**: king − man + woman ≈ queen.
- Uses: semantic search, RAG, recommendations, clustering, duplicates.

**Embeddings in context.** The *lookup* for "bank" is identical in "river bank" and "money … bank". After 28 layers the two vectors differ (cosine 0.79): attention has mixed in the neighbours. Reading the intermediate vectors through the final layer ("logit lens"), "The cat says meow. The cow says …" only turns into `·moo` in the last layers (layer 25 still says `·cow`, layer 28: 93.5% `·moo`).

> [!example] Exam question 1 (from the example sheet) in this light
> "the ·river ·bank" vs "the ·savings ·bank": **same embedding** (lookup is static) ✓; **after attention** bank holds meaning from earlier words ✓; after positional encoding the two banks are **not** different (both position 3); the feed-forward does **not** mix neighbours (it sees one token at a time).

> [!example]+ Slides 39, 43, 44, 46, 49
> ![[GenAI-L1-slajd39.png|760]]
>
> ![[GenAI-L1-slajd43.png|760]]
>
> ![[GenAI-L1-slajd44.png|760]]
>
> ![[GenAI-L1-slajd46.png|760]]
>
> ![[GenAI-L1-slajd49.png|760]]

---

## Exam traps from this lecture

- Generative = learns $p(x)$; discriminative = $p(y\mid x)$. A generative model *can* also classify (via Bayes), a discriminative one cannot generate.
- The chain rule is exact; "autoregressive" ≠ "approximation".
- In diffusion training, the network predicts the **noise**, the noise itself is drawn, not learned.
- BPE starts from **bytes** (256), so there are no unknown tokens.
- Symbolic systems fail on cases outside their rules; subsymbolic systems can only be corrected indirectly.

Related: [[GenAI - Glossary|Glossary: symbols & variables]] · [[2026-10-28 - Theoretical Foundations of Language Modelling|Theoretical Foundations of Language Modelling]] · [[GenAI - Mock Test A|Mock Test A]]
