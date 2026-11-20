---
publish: true
tags: [genai, lecture, test1]
lecture: 2026-11-04
lecturer: Johannes Oster
site: https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/
course: Generative AI (194.207)
---

# 03 · Training Paradigms I (4 Nov) – architecture, data, pre-training

> [!abstract] The story
> - We want to build **ShallowSeek**: a frontier-style LLM that is (1) multilingual, (2) multimodal (text + images), (3) has a 1M-token context.
> - We **optimise for evals** by choosing **1 architecture · 2 data · 3 pre-training · 4 mid-training · 5 post-training · 6 deployment**, **subject to constraints** (budget!).
> - Method: *start small · measure · scale up · repeat*.
> Budget: a grant of **130,000 H100-hours** (≈ 30 H100s for 6 months). Split: ~50k final run (pilot pre-training ~40k + mid/post), ~10k data work (classifiers, OCR, captions), ~50k experiments, ~20k evaluation/failures/buffer.
> This lecture covers 1–3. Training Paradigms II covers 4–5.

---

## Interactive page (5w1h)

<iframe src="https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/" width="100%" height="750" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!tip]- Section links
> [Architecture](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:architecture) · [Scaling laws](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#scaling-laws) · [Compute optimality](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#compute-optimality) · [Step 1 compute](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:compute) · [Step 2 params & tokens](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:params-tokens) · [Step 3 width & depth](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:width-depth) · [Step 4 experts](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:experts) · [Tokenizer cost](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:tokenizer-cost) · [Vocab cost](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:vocab-cost) · [Images](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:images) · [Attention cost](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:attention-cost) · [KV cache](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:kv-cache) · [GQA](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:gqa) · [MLA](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:mla) · [Data](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:data) · [Decontamination](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:decontamination) · [Packing](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:packing) · [Images as tokens](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:image-tokens) · [Loss](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:loss) · [Backward](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:backward) · [Parallelism](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:parallel) · [Update](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:update)
>
> The site's chapter also contains the long-context material (KV cache, GQA, MLA) that the **slides** moved to Training Paradigms II → see [[2026-11-11 - Training Paradigms II|Training Paradigms II]]. For calculators (compute budget, Chinchilla split, vocabulary share, loss) use the **Compute** tab of [[GenAI Lab]].

---

## 1 · Architecture – what can be chosen

| Knob | Options | Pilot's choice |
|---|---|---|
| Tokenizer | vocabulary size, language support | ~150k, multilingual |
| Images | added later (adapter) or trained in; own encoder or none | (see §1.6) |
| Width | embedding dimension $d$, number of heads | $d$ = 2,048 = 16 heads × 128 |
| Depth | number of blocks $L$ | 24 |
| Feed-forward | width | experts 768 wide; 9 active = 6,912 = 3.4·d |
| Sparsity | dense or MoE; #experts; experts per token; active vs total | 8 of 256 routed + 1 shared; **2B active of 30B total** |
| Attention | KV sharing (GQA, MLA), window | → TP II |
| RoPE | context length | → TP II |

> [!example]- Slide 34 (click to open)
> ![[GenAI-L3-slajd34.png|720]]

### 1.1 Scaling laws (Kaplan et al. 2020)

Loss falls as a **power law** (a straight line on log–log axes) in compute $C$, data $D$ and non-embedding parameters $N$, as long as the other two don't bottleneck. Bigger models are more **sample-efficient**: they reach a given loss with fewer tokens.

> [!example]- Slide 38 (click to open)
> ![[GenAI-L3-slajd38.png|720]]

### 1.2 Compute-optimal (Chinchilla, Hoffmann et al. 2022)

For a fixed compute budget, the best loss comes from scaling $N$ and $D$ **about equally** ($N_{opt}\propto C^{0.49}$, $D_{opt}\propto C^{0.51}$) → rule of thumb **≈ 20 tokens per parameter**. (Dense models.)

> [!example]- Slide 39 (click to open)
> ![[GenAI-L3-slajd39.png|720]]

### 1.3 Where C ≈ 6·N·D comes from

One weight $w$, one token. A FLOP = one multiply **or** one add.

| Pass | Operation | FLOPs |
|---|---|---|
| forward | output: $y \mathrel{+}= w\cdot x$ | 2 |
| backward | gradient for the **input**: $\partial x \mathrel{+}= w\cdot\partial y$ | 2 |
| backward | gradient for the **weight**: $\partial w \mathrel{+}= x\cdot\partial y$ | 2 |
| | **per weight, per token** | **6** |

$$C\approx 6\cdot N\cdot D$$

- $N$ = parameters a token is multiplied with: in a **MoE only the active** ones; embeddings are a lookup (no multiply) so Kaplan counts **non-embedding** parameters.
- $D$ = training tokens.
- Left out: **attention's** score computation ($QK^\top$, $\alpha V$) and the optimiser step (once per *step*, not per token).
- Backward = 2× forward because each weight is used for **two** gradients (input's and weight's).

With Chinchilla $D\approx20N$: $C\approx120N^2 \Rightarrow N\approx\sqrt{C/120}$.

> [!example]- Slide 41 (click to open)
> ![[GenAI-L3-slajd41.png|720]]

### 1.4 The pilot's numbers, step by step

| Step | Computation | Result |
|---|---|---|
| FLOPs per H100-hour at bf16 peak | $989\cdot10^{12}\ \text{FLOP/s}\times3600\ \text{s}$ | $3.56\cdot10^{18}$ |
| realistic (MFU ≈ 35%) | $3.56\cdot10^{18}\times0.35$ | $1.23\cdot10^{18}$ |
| pilot pre-training | $40{,}000\times1.23\cdot10^{18}$ | $C=4.9\cdot10^{22}$ FLOPs (31% of grant) |
| compute-optimal size | $\sqrt{4.9\cdot10^{22}/120}$ | $N\approx20$B on $D\approx0.4$T |
| **chosen instead** | $6\cdot2\cdot10^9\cdot4\cdot10^{12}$ | **2B on 4T** (≈ 2,000 tokens/param) |

> [!important] Why deliberately *not* compute-optimal?
> 2B on 4T has only **~4% higher loss** than the optimal 20B on 0.4T (the loss curve is flat near the optimum: 40B +0.3%, 3.6B +2.3%, 2B +4.1%, 1B +7.4%), but the model is **10× smaller**, so everything after pre-training (mid/post-training experiments, RL rollouts, inference, KV cache) **costs a tenth**. Chinchilla optimises *training* compute only; real projects pay for inference too → "over-train" small models.

**MFU** = model FLOPs utilisation: the fraction of the GPU's peak that does useful model math (rest: communication, memory waits, idle).

**Try it: Compute budget: C ≈ 6·N·D**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=compute" width="100%" height="720" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slides 40, 42 (click to open)
> ![[GenAI-L3-slajd40.png|720]]
>
> ![[GenAI-L3-slajd42.png|720]]

### 1.5 Width vs depth

Loss is nearly flat across shapes as long as **d ÷ blocks ≈ 20–250** (Kaplan Fig. 5). For 2B active and a ~150k vocabulary: 1,024 × 116 blocks (too deep), **2,048 × 24 (chosen, ≈ 85)**, 4,096 × 3 (too wide). $d$ = heads × 128 → 16 heads.

> [!example]- Slide 44 (click to open)
> ![[GenAI-L3-slajd44.png|720]]

### 1.6 Dense vs mixture of experts

| | Dense | MoE (pilot) |
|---|---|---|
| FFN per token | one FFN | router picks experts; + 1 **shared** expert always on |
| Active / total | 2B / 2B | **2B / 30B** |
| Design steps | | ① how wide per token: like a dense model's FFN, ≈ 3.4·d = 6,912 ② cut into experts of 0.375·d = **768** → 9 = 1 shared + 8 routed ③ how many in total: keep **32×** as many → 256 routed |

The trend 2024→2026: **finer** (smaller experts) and **sparser** (higher total ÷ active). $C\approx6ND$ uses **active** $N$: MoE adds knowledge capacity without adding per-token compute (but adds memory and communication).

> [!example]- Slides 45, 46 (click to open)
> ![[GenAI-L3-slajd45.png|720]]
>
> ![[GenAI-L3-slajd46.png|720]]

### 1.7 Tokenizer – what a language costs

**Fertility** = how many tokens a text needs. The same document (UDHR) costs Greek **7.2×**, Bulgarian 6.3×, Polish 3.1×, German 2.3×, French 2.0× as many tokens as English with **GPT-2's English-trained 50k** tokenizer. Better with Qwen3 (152k) and EuroLLM (128k, learned on EU languages). More tokens = more compute per sentence, more cost per API call, less text fits in the context, slower generation → **a multilingual model needs a multilingual tokenizer**.

> [!example]- Slide 49 (click to open)
> ![[GenAI-L3-slajd49.png|720]]

### 1.8 What a vocabulary costs

Embedding ($V\times d$) + output layer ($d\times V$) = **$2\cdot V\cdot d$** weights (untied).

| $V$ | share of pilot (d 2,048, 2B active) | share of frontier (d 7,168, 70B active) |
|---|---|---|
| 50k (GPT-2) | 10% | 1.0% |
| 128k (Llama 3, EuroLLM) | 26% | 2.6% |
| **150k** (Qwen3 152k, Kimi K3 164k) | **30%** | 3.1% |
| 256k (Gemma 3 262k) | 51% | 5.2% |

Small models pay heavily for big vocabularies; pilot picks ~150k as the multilingual compromise.

> [!example]- Slide 50 (click to open)
> ![[GenAI-L3-slajd50.png|720]]

### 1.9 Images

| Route | Example | + | − |
|---|---|---|---|
| **Adapter**: separately trained vision encoder + projector into a finished LM | Qwen2.5-VL, most open models | cheap, reuses two finished models | images join a model built for text |
| **Native, with encoder**: encoder + LM trained together | Kimi K3 | text and images learn together | one large joint run from scratch |
| **Native, without encoder**: patches → linear embedding straight into the LM | Fuyu-8B | simplest: one network | less proven at scale |

Timing (Kimi K2.5): **early, a little at a time** (10% images from start to end) beats mid-way (20% in second half) and late (50% in last fifth) on **both** image and text scores.

> [!example]- Slides 51, 52 (click to open)
> ![[GenAI-L3-slajd51.png|720]]
>
> ![[GenAI-L3-slajd52.png|720]]

---

## 2 · Data

Knobs: **amount** (how many tokens, do they exist?), **scope** (languages, inputs), **sources**, **cleaning** (filters, dedup, quality classifier), **decontamination**, **mix** (share per source & language), **repeats** (upsampling, rephrasing), **images** (resolution, captions), **tokenizer** (vocab, fertility).

- **Sources:** Common Crawl (~100 monthly crawls; ~240T tokens as text) → curated sets FineWeb 15T, DCLM-Baseline 4T, FineWeb-Edu 1.3T, Nemotron-CC 6.3T, FineWeb-2 (multilingual); from the source: code (342B), papers/PDFs, arXiv LaTeX, Gutenberg, Wikipedia; image links. *Not crawled:* opt-outs, logins, paywalls, social media, video, audio.
- **Cleaning:** heuristic filters (junk like `#$%@&!`) → deduplicate (exact copies, near copies, repeated passages) → classify (topic and quality).
- **Decontamination:** remove pages that contain benchmark items (e.g. a forum post containing a GSM8K problem with its answer: *same words, same order* → dropped). Otherwise the eval measures memorisation.
- **The mix:** train many small proxy models on different mixes, fit a predictor of score from shares, pick the predicted best under constraints (e.g. no source repeated > 4×).
- **Packing:** end each document with `<|endoftext|>` (ID 151643), join everything, cut into fixed rows. Padding short docs wastes ~44%; packing wastes 0%.
- **Images as tokens (Qwen2.5-VL):** 14×14-px patches → vision encoder → one vector per **2×2 patches** = one `<|image_pad|>` slot. A 448×448 image = 32×32 = 1,024 patches = **256 slots**, between `<|vision_start|>` and `<|vision_end|>`. The slot embeddings are overwritten with the image vectors (`masked_scatter`); their labels are set to **-100** → **no loss**: images are input only, not predicted.

> [!example]- Slides 56, 57, 58, 59, 61, 62 (click to open)
> ![[GenAI-L3-slajd56.png|720]]
>
> ![[GenAI-L3-slajd57.png|720]]
>
> ![[GenAI-L3-slajd58.png|720]]
>
> ![[GenAI-L3-slajd59.png|720]]
>
> ![[GenAI-L3-slajd61.png|720]]
>
> ![[GenAI-L3-slajd62.png|720]]

---

## 3 · Pre-training – one step

① next batch from the packed dataset → ② forward → ③④ loss → ⑤ update → ⑥ evaluate (now and then).

| Thing | Value |
|---|---|
| batch | 1,024 rows × 4,096 tokens ≈ **4.2M tokens per step** |
| input / target | the **same row shifted by one**: `inputs = batch[:, :-1]`, `targets = batch[:, 1:]` |
| forward shapes | token IDs (rows × tokens) → embeddings (rows × tokens × d) → 24 blocks → logits (rows × tokens × V) → probabilities |
| loss per token | $-\log p(\text{true next token})$: small $p$ → big loss (p = 0.033 → 3.41; p = 0.000024 → 10.7) |
| batch loss | **mean** over all ≈4.2M token losses → **one number** $\mathcal{L}$ |
| many GPUs | each GPU: mean over its rows; batch loss = **mean of means** (equal counts) |
| backward | from that one number: a gradient for every weight |
| loss spikes | a rare batch meets an unstable state → huge gradient → **clip** the global gradient norm to 1.0 (same direction, shorter); normal steps untouched |
| update | $\theta\leftarrow\theta-\eta_t\cdot\Delta_t$ |

**Update rule variables.** $\theta$ = every weight. $\eta_t$ = learning rate at step $t$, set by the **scheduler** (warmup: start gently → plateau: fast progress → decay: settle; e.g. cosine, warmup–stable–decay (WSD), linear decay). $\Delta_t$ = step from the **optimiser**: one batch's gradient is noisy and badly scaled, so smooth it (momentum) and size it per weight (AdamW; Muon orthogonalises the update matrix).

> [!example]- Slides 66, 71, 75, 78 (click to open)
> ![[GenAI-L3-slajd66.png|720]]
>
> ![[GenAI-L3-slajd71.png|720]]
>
> ![[GenAI-L3-slajd75.png|720]]
>
> ![[GenAI-L3-slajd78.png|720]]

### Parallelism – what travels between GPUs

| Strategy | Split | What travels |
|---|---|---|
| **Data parallel** | copy the model, split the rows | gradients, averaged (all-reduce) once per step |
| **FSDP / ZeRO-3** | split the *storage* of weights/grads/optimiser state; compute all | weights & gradients: each layer gathered, then freed |
| **Tensor parallel** | split every layer's matrices | partial results, summed after every layer |
| **Pipeline parallel** | split by layers (GPU1 = first layers …) | activations forward, their gradients back |
| **Expert parallel** | split the experts | tokens, to their experts and back |

```python
opt = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
for step, batch in enumerate(loader):
    inputs = batch[:, :-1]; targets = batch[:, 1:].clone()
    targets[targets == 151655] = -100          # image slots: no loss
    logits = model(inputs)
    loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    set_lr(opt, schedule(step))
    opt.step(); opt.zero_grad()
```

> [!example]- Slide 77 (click to open)
> ![[GenAI-L3-slajd77.png|720]]

---

## Practice questions (8)

> [!info] Same format as the test
> 0 to 4 options can be correct: decide each option on its own, true or false. Next to each option is the section above that answers it.
> Full timed tests: [[GenAI - Mock Test A|Mock A]] · [[GenAI - Mock Test B|Mock B]] · [[GenAI - Mock Test C|Mock C]] · [[GenAI - Mock Test D|Mock D]] (their questions are not repeated here).

### P1 · where C ≈ 6 · N · D comes from
<small>from the official example sheet</small>

**The compute for training is about C ≈ 6 · N · D. Where does the 6 come from?**

- [ ] **A** Forward pass: each weight does one multiply and one add per token <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **B** Backward pass: each weight is used twice, for two different gradients <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **C** Optimizer step: two more operations for each weight and token <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **D** Attention: two more operations for comparing the tokens <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> Forward: 2 operations for each weight and token. Backward: 4. Together: 6. The optimizer runs once per step, not once per token, and the formula leaves attention out.

### P2 · width vs depth

**The pilot has 2B active parameters, d = 2,048, 24 blocks and 16 heads. What is true?**

- [ ] **A** Each head has 128 dimensions <small>· look in [[2026-11-04 - Training Paradigms I#1 · Architecture – what can be chosen|TP I §1 Architecture]]</small>
- [ ] **B** d ÷ blocks ≈ 85, inside the good range of 20–250 <small>· look in [[2026-11-04 - Training Paradigms I#1.5 Width vs depth|TP I §1.5 Width vs depth]]</small>
- [ ] **C** 4,096 × 3 blocks would be too deep <small>· look in [[2026-11-04 - Training Paradigms I#1.5 Width vs depth|TP I §1.5 Width vs depth]]</small>
- [ ] **D** Inside that range, loss changes by only about 1 % <small>· look in [[2026-11-04 - Training Paradigms I#1.5 Width vs depth|TP I §1.5 Width vs depth]]</small>

> [!success]- Answer: A, B, D
> A ✓ · B ✓ · C ✗ · D ✓
>
> 2,048 / 24 ≈ 85; 2,048 / 16 = 128. 4,096 × 3 is too wide, 1,024 × 116 too deep (Kaplan Fig. 5, EuroLLM-1.7B’s shape).

### P3 · tokenizer fertility

**The same text needs 7.2× as many tokens in Greek as in English with GPT-2’s tokenizer. What is true?**

- [ ] **A** Less Greek text fits into a fixed context window <small>· look in [[2026-11-04 - Training Paradigms I#1.7 Tokenizer – what a language costs|TP I §1.7 Tokenizer]]</small>
- [ ] **B** Greek text costs more compute per sentence <small>· look in [[2026-11-04 - Training Paradigms I#1.7 Tokenizer – what a language costs|TP I §1.7 Tokenizer]]</small>
- [ ] **C** A tokenizer trained on EU languages reduces the factor <small>· look in [[2026-11-04 - Training Paradigms I#1.7 Tokenizer – what a language costs|TP I §1.7 Tokenizer]]</small>
- [ ] **D** Training longer removes the problem <small>· look in [[2026-11-04 - Training Paradigms I#1.7 Tokenizer – what a language costs|TP I §1.7 Tokenizer]]</small>

> [!success]- Answer: A, B, C
> A ✓ · B ✓ · C ✓ · D ✗
>
> Fertility is a property of the tokenizer, fixed before training. More tokens per sentence means more compute, higher API cost, shorter effective context and slower generation.

### P4 · packing

**Documents are packed into training rows of 4,096 tokens. What is true?**

- [ ] **A** A document longer than 4,096 tokens is dropped <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **B** Packing wastes less compute than padding each document to the row length <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **C** Every document must be exactly 4,096 tokens long <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **D** Every document is followed by <|endoftext|> <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>

> [!success]- Answer: B, D
> A ✗ · B ✓ · C ✗ · D ✓
>
> Join all documents, each ended with <|endoftext|> (151643), and cut into rows: 0 % waste vs 44 % with padding. A long document is cut across rows.

### P5 · optimiser and schedule

**The update is θ ← θ − ηₜ · Δₜ. What is true?**

- [ ] **A** With AdamW, Δₜ is simply the raw gradient of the current batch <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **B** In a warmup–stable–decay schedule the learning rate never changes <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **C** ηₜ comes from the learning-rate schedule, e.g. warmup, plateau, decay <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **D** AdamW smooths the gradient and sizes the step per weight <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>

> [!success]- Answer: C, D
> A ✗ · B ✗ · C ✓ · D ✓
>
> Optimiser (Δ): one batch’s gradient is noisy and badly scaled → smooth it (momentum) and size it per weight (AdamW, Muon). Scheduler (η): start gently, go big, end small.

### P6 · images in the architecture

**Ways to add images to a language model. What is true?**

- [ ] **A** In the adapter route, text and images learn together from the start <small>· look in [[2026-11-04 - Training Paradigms I#1.9 Images|TP I §1.9 Images]]</small>
- [ ] **B** A native model without an encoder is the most proven route among frontier models <small>· look in [[2026-11-04 - Training Paradigms I#1.9 Images|TP I §1.9 Images]]</small>
- [ ] **C** Adding images late in training (50 % in the last fifth) scored best in Kimi K2.5’s comparison <small>· look in [[2026-11-04 - Training Paradigms I#1.9 Images|TP I §1.9 Images]]</small>
- [ ] **D** An adapter joins a separately trained vision encoder to a finished language model through a projector <small>· look in [[2026-11-04 - Training Paradigms I#1.9 Images|TP I §1.9 Images]]</small>

> [!success]- Answer: D
> A ✗ · B ✗ · C ✗ · D ✓
>
> Adapter (Qwen2.5-VL): cheap, but images join a model built for text. Native with encoder (Kimi K3): learn together, one big joint run. Native without encoder (Fuyu-8B): simplest, less proven. Early, a little at a time (10 %) was best on images and on text.

### P7 · scaling laws (Kaplan)

**What is true about the scaling laws of Kaplan et al. (2020)?**

- [ ] **A** Bigger models reach a given loss with fewer tokens <small>· look in [[2026-11-04 - Training Paradigms I#1.1 Scaling laws (Kaplan et al. 2020)|TP I §1.1 Scaling laws]]</small>
- [ ] **B** They count non-embedding parameters <small>· look in [[2026-11-04 - Training Paradigms I#1.1 Scaling laws (Kaplan et al. 2020)|TP I §1.1 Scaling laws]]</small>
- [ ] **C** Loss falls in a straight line against compute on normal (linear) axes <small>· look in [[2026-11-04 - Training Paradigms I#1.1 Scaling laws (Kaplan et al. 2020)|TP I §1.1 Scaling laws]]</small>
- [ ] **D** The power law holds even when data is the bottleneck <small>· look in [[2026-11-04 - Training Paradigms I#1.1 Scaling laws (Kaplan et al. 2020)|TP I §1.1 Scaling laws]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> A power law is a straight line on log–log axes. It holds as long as neither of the other two factors bottlenecks. Bigger models are more sample-efficient.

### P8 · FSDP and pipeline parallel

**What is true about splitting training over GPUs?**

- [ ] **A** FSDP / ZeRO-3 splits the storage of weights, gradients and optimiser state <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>
- [ ] **B** Pipeline parallel sends activations forward and their gradients back <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>
- [ ] **C** FSDP never moves weights between GPUs <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>
- [ ] **D** Data parallel gives each GPU a different part of the model <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> FSDP gathers each layer’s weights when needed and frees them again. Data parallel copies the whole model and splits the rows.

---

## Exam traps from this lecture

- 6 = 2 (forward) + 4 (backward: two gradients × multiply+add). Optimiser and attention are **not** in the 6.
- $N$ in $6ND$ = **active** parameters for MoE.
- Chinchilla: ~20 tokens/param minimises loss **for fixed training compute**, not total cost. Overtraining a small model is a deliberate trade.
- Vocabulary cost $2Vd$ matters most for **small** models.
- Packing + `<|endoftext|>`; images: `-100` labels = no loss.
- Loss is the **mean** $-\log p$ of the true next token; one scalar for the whole batch.
- Gradient clipping rescales the **whole** gradient to length 1.0 only when it is longer; direction unchanged.
- Data parallel moves **gradients**; tensor parallel moves **partial results**; pipeline moves **activations**; expert parallel moves **tokens**.

Related: [[GenAI - Glossary|Glossary: symbols & variables]] · [[2026-10-28 - Theoretical Foundations of Language Modelling|Theoretical Foundations of Language Modelling]] · [[2026-11-11 - Training Paradigms II|Training Paradigms II]] · [[GenAI Lab]]
