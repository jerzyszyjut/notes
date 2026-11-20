---
publish: true
tags: [genai, lecture, test1]
lecture: 2026-11-04
lecturer: Johannes Oster
site: https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/
course: Generative AI (194.207)
---

# 03 · Training Paradigms I (4 Nov): architecture, data, pre-training

> [!abstract] The story
> We want to build **ShallowSeek**: a frontier-style LLM that is (1) multilingual, (2) multimodal (text + images), (3) has a 1M-token context. We **optimise for evals** by choosing **1 architecture · 2 data · 3 pre-training · 4 mid-training · 5 post-training · 6 deployment**, **subject to constraints** (budget!). Method: *start small · measure · scale up · repeat*.
> Budget: a grant of **130,000 H100-hours** (≈ 30 H100s for 6 months). Split: ~50k final run (pilot pre-training ~40k + mid/post), ~10k data work (classifiers, OCR, captions), ~50k experiments, ~20k evaluation/failures/buffer.
> This lecture covers 1–3. Training Paradigms II covers 4–5.

## Interactive page (5w1h)

<iframe src="https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/" width="100%" height="750" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!tip] Section links
> [Architecture](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:architecture) · [Scaling laws](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#scaling-laws) · [Compute optimality](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#compute-optimality) · [Step 1 compute](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:compute) · [Step 2 params & tokens](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:params-tokens) · [Step 3 width & depth](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:width-depth) · [Step 4 experts](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:experts) · [Tokenizer cost](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:tokenizer-cost) · [Vocab cost](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:vocab-cost) · [Images](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:images) · [Attention cost](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:attention-cost) · [KV cache](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:kv-cache) · [GQA](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:gqa) · [MLA](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:mla) · [Data](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:data) · [Decontamination](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:decontamination) · [Packing](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:packing) · [Images as tokens](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:image-tokens) · [Loss](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:loss) · [Backward](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:backward) · [Parallelism](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:parallel) · [Update](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:update)
>
> The site's chapter also contains the long-context material (KV cache, GQA, MLA) that the **slides** moved to Training Paradigms II → see [[2026-11-11 - Training Paradigms II|Training Paradigms II]]. For calculators (compute budget, Chinchilla split, vocabulary share, loss) use the **Compute** tab of [[GenAI Lab]].

---

## 1 · Architecture: what can be chosen

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

> [!example]+ Slide 34
> ![[GenAI-L3-slajd34.png|760]]

### 1.1 Scaling laws (Kaplan et al. 2020)

Loss falls as a **power law** (a straight line on log–log axes) in compute $C$, data $D$ and non-embedding parameters $N$, as long as the other two don't bottleneck. Bigger models are more **sample-efficient**: they reach a given loss with fewer tokens.

> [!example]+ Slide 38
> ![[GenAI-L3-slajd38.png|760]]

### 1.2 Compute-optimal (Chinchilla, Hoffmann et al. 2022)

For a fixed compute budget, the best loss comes from scaling $N$ and $D$ **about equally** ($N_{opt}\propto C^{0.49}$, $D_{opt}\propto C^{0.51}$) → rule of thumb **≈ 20 tokens per parameter**. (Dense models.)

> [!example]+ Slide 39
> ![[GenAI-L3-slajd39.png|760]]

### 1.3 Where $C\approx 6ND$ comes from

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

> [!example]+ Slide 41
> ![[GenAI-L3-slajd41.png|760]]

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

> [!example]+ Slides 40, 42
> ![[GenAI-L3-slajd40.png|760]]
>
> ![[GenAI-L3-slajd42.png|760]]

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm#compute" width="100%" height="860" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

*Interactive: **Compute** tab of the [[GenAI Lab]] · [[genai-lab.htm|open it full screen]]*

### 1.5 Width vs depth

Loss is nearly flat across shapes as long as **d ÷ blocks ≈ 20–250** (Kaplan Fig. 5). For 2B active and a ~150k vocabulary: 1,024 × 116 blocks (too deep), **2,048 × 24 (chosen, ≈ 85)**, 4,096 × 3 (too wide). $d$ = heads × 128 → 16 heads.

> [!example]+ Slide 44
> ![[GenAI-L3-slajd44.png|760]]

### 1.6 Dense vs mixture of experts

| | Dense | MoE (pilot) |
|---|---|---|
| FFN per token | one FFN | router picks experts; + 1 **shared** expert always on |
| Active / total | 2B / 2B | **2B / 30B** |
| Design steps | | ① how wide per token: like a dense model's FFN, ≈ 3.4·d = 6,912 ② cut into experts of 0.375·d = **768** → 9 = 1 shared + 8 routed ③ how many in total: keep **32×** as many → 256 routed |

The trend 2024→2026: **finer** (smaller experts) and **sparser** (higher total ÷ active). $C\approx6ND$ uses **active** $N$: MoE adds knowledge capacity without adding per-token compute (but adds memory and communication).

> [!example]+ Slides 45, 46
> ![[GenAI-L3-slajd45.png|760]]
>
> ![[GenAI-L3-slajd46.png|760]]

### 1.7 Tokenizer: what a language costs

**Fertility** = how many tokens a text needs. The same document (UDHR) costs Greek **7.2×**, Bulgarian 6.3×, Polish 3.1×, German 2.3×, French 2.0× as many tokens as English with **GPT-2's English-trained 50k** tokenizer. Better with Qwen3 (152k) and EuroLLM (128k, learned on EU languages). More tokens = more compute per sentence, more cost per API call, less text fits in the context, slower generation → **a multilingual model needs a multilingual tokenizer**.

> [!example]+ Slide 49
> ![[GenAI-L3-slajd49.png|760]]

### 1.8 What a vocabulary costs

Embedding ($V\times d$) + output layer ($d\times V$) = **$2\cdot V\cdot d$** weights (untied).

| $V$ | share of pilot (d 2,048, 2B active) | share of frontier (d 7,168, 70B active) |
|---|---|---|
| 50k (GPT-2) | 10% | 1.0% |
| 128k (Llama 3, EuroLLM) | 26% | 2.6% |
| **150k** (Qwen3 152k, Kimi K3 164k) | **30%** | 3.1% |
| 256k (Gemma 3 262k) | 51% | 5.2% |

Small models pay heavily for big vocabularies; pilot picks ~150k as the multilingual compromise.

> [!example]+ Slide 50
> ![[GenAI-L3-slajd50.png|760]]

### 1.9 Images

| Route | Example | + | − |
|---|---|---|---|
| **Adapter**: separately trained vision encoder + projector into a finished LM | Qwen2.5-VL, most open models | cheap, reuses two finished models | images join a model built for text |
| **Native, with encoder**: encoder + LM trained together | Kimi K3 | text and images learn together | one large joint run from scratch |
| **Native, without encoder**: patches → linear embedding straight into the LM | Fuyu-8B | simplest: one network | less proven at scale |

Timing (Kimi K2.5): **early, a little at a time** (10% images from start to end) beats mid-way (20% in second half) and late (50% in last fifth) on **both** image and text scores.

> [!example]+ Slides 51, 52
> ![[GenAI-L3-slajd51.png|760]]
>
> ![[GenAI-L3-slajd52.png|760]]

## 2 · Data

Knobs: **amount** (how many tokens, do they exist?), **scope** (languages, inputs), **sources**, **cleaning** (filters, dedup, quality classifier), **decontamination**, **mix** (share per source & language), **repeats** (upsampling, rephrasing), **images** (resolution, captions), **tokenizer** (vocab, fertility).

- **Sources:** Common Crawl (~100 monthly crawls; ~240T tokens as text) → curated sets FineWeb 15T, DCLM-Baseline 4T, FineWeb-Edu 1.3T, Nemotron-CC 6.3T, FineWeb-2 (multilingual); from the source: code (342B), papers/PDFs, arXiv LaTeX, Gutenberg, Wikipedia; image links. *Not crawled:* opt-outs, logins, paywalls, social media, video, audio.
- **Cleaning:** heuristic filters (junk like `#$%@&!`) → deduplicate (exact copies, near copies, repeated passages) → classify (topic and quality).
- **Decontamination:** remove pages that contain benchmark items (e.g. a forum post containing a GSM8K problem with its answer: *same words, same order* → dropped). Otherwise the eval measures memorisation.
- **The mix:** train many small proxy models on different mixes, fit a predictor of score from shares, pick the predicted best under constraints (e.g. no source repeated > 4×).
- **Packing:** end each document with `<|endoftext|>` (ID 151643), join everything, cut into fixed rows. Padding short docs wastes ~44%; packing wastes 0%.
- **Images as tokens (Qwen2.5-VL):** 14×14-px patches → vision encoder → one vector per **2×2 patches** = one `<|image_pad|>` slot. A 448×448 image = 32×32 = 1,024 patches = **256 slots**, between `<|vision_start|>` and `<|vision_end|>`. The slot embeddings are overwritten with the image vectors (`masked_scatter`); their labels are set to **-100** → **no loss**: images are input only, not predicted.

> [!example]+ Slides 56, 57, 58, 59, 61, 62
> ![[GenAI-L3-slajd56.png|760]]
>
> ![[GenAI-L3-slajd57.png|760]]
>
> ![[GenAI-L3-slajd58.png|760]]
>
> ![[GenAI-L3-slajd59.png|760]]
>
> ![[GenAI-L3-slajd61.png|760]]
>
> ![[GenAI-L3-slajd62.png|760]]

## 3 · Pre-training: one step

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

> [!example]+ Slides 66, 71, 75, 78
> ![[GenAI-L3-slajd66.png|760]]
>
> ![[GenAI-L3-slajd71.png|760]]
>
> ![[GenAI-L3-slajd75.png|760]]
>
> ![[GenAI-L3-slajd78.png|760]]

### Parallelism: what travels between GPUs

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

> [!example]+ Slide 77
> ![[GenAI-L3-slajd77.png|760]]

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


## Test questions for this lecture (18)

> [!info] Same format as the test
> 0 to 4 options can be correct. Tick, then open the answer. **+1.5 ÷ n** per correct tick, **−0.75** per wrong tick, never below 0; no correct option → no tick = 1.5.
> The same questions, mixed across lectures and timed, are in the mock tests ([[_GenAI Test 1 - Start Here|Start Here]]) and in the Quiz tab of the [[GenAI Lab]].

### Official Q3 · where C ≈ 6 · N · D comes from
<small>from [[GenAI - Official Example Questions|Official]]</small>

**The compute for training is about C ≈ 6 · N · D. Where does the 6 come from?**

- [ ] **A** Forward pass: each weight does one multiply and one add per token
- [ ] **B** Backward pass: each weight is used twice, for two different gradients
- [ ] **C** Optimizer step: two more operations for each weight and token
- [ ] **D** Attention: two more operations for comparing the tokens

> [!success]- Answer: A, B
> Forward: 2 operations for each weight and token. Backward: 4. Together: 6. The optimizer runs once per step, not once per token, and the formula leaves attention out.
> *Training Paradigms I · where C ≈ 6 · N · D comes from*

### Mock A Q9 · C ≈ 6·N·D in a MoE
<small>from [[GenAI - Mock Test A|Mock A]]</small>

**A mixture-of-experts model has 30B parameters in total, 2B active per token, and is trained on 4T tokens. What is true?**

- [ ] **A** The FLOPs for computing QKᵀ are not part of 6 · N · D
- [ ] **B** The training compute is about 6 · 30B · 4T, since every expert exists in memory
- [ ] **C** The training compute is about 6 · 2B · 4T ≈ 4.8 · 10²² FLOPs
- [ ] **D** The 6 includes two operations per weight for the optimizer step

> [!success]- Answer: A, C
> N counts the parameters a token is multiplied with: in a MoE, the active ones. 6 = 2 forward + 4 backward per weight and token; the optimizer runs once per step, not per token. The formula leaves attention scores out.
> *Training Paradigms I · C ≈ 6·N·D in a MoE*

### Mock A Q10 · compute-optimal vs over-trained
<small>from [[GenAI - Mock Test A|Mock A]]</small>

**The pilot has C ≈ 4.9 · 10²² FLOPs. Chinchilla-optimal would be about 20B parameters on 0.4T tokens; the pilot trains 2B on 4T instead. What is true?**

- [ ] **A** The 2B model reaches a lower loss than the 20B model
- [ ] **B** Chinchilla’s rule of thumb is about 20 parameters per token
- [ ] **C** Both choices need roughly the same training compute
- [ ] **D** Everything after pre-training (fine-tuning, RL, inference) becomes roughly ten times cheaper

> [!success]- Answer: C, D
> 6 · 20B · 0.4T = 6 · 2B · 4T = 4.8 · 10²². The smaller model pays about +4 % loss but is 10× smaller, so all later stages cost a tenth. Chinchilla: about 20 tokens per parameter.
> *Training Paradigms I · compute-optimal vs over-trained*

### Mock A Q11 · vocabulary cost
<small>from [[GenAI - Mock Test A|Mock A]]</small>

**The pilot has d = 2,048 and 2B active parameters; embedding and output layer are separate. What is true for a vocabulary of V = 150k?**

- [ ] **A** Embedding plus output layer hold about 2 · V · d ≈ 0.6B weights
- [ ] **B** A smaller vocabulary always makes training on Greek text cheaper
- [ ] **C** That is roughly 30 % of the active parameters
- [ ] **D** For a frontier model with d = 7,168 and 70B active, the same vocabulary is only about 3 %

> [!success]- Answer: A, C, D
> 2 · 150k · 2,048 ≈ 0.61B, ≈ 30 % of 2B; 2 · 150k · 7,168 ≈ 2.2B ≈ 3 % of 70B. A smaller or English-centred vocabulary splits Greek into many more tokens (7.2× with GPT-2’s), which costs more compute per sentence.
> *Training Paradigms I · vocabulary cost*

### Mock A Q12 · images as tokens
<small>from [[GenAI - Mock Test A|Mock A]]</small>

**In Qwen2.5-VL-style training, an image appears as a run of <|image_pad|> slots between <|vision_start|> and <|vision_end|>. What is true?**

- [ ] **A** The labels at the slot positions are set to −100, so they get no loss
- [ ] **B** A 448 × 448 image in 14 × 14 patches, merged 2 × 2 per slot, gives 256 slots
- [ ] **C** The model learns to predict the image patches as its next tokens
- [ ] **D** The vision encoder’s vectors overwrite the embeddings at the slot positions

> [!success]- Answer: A, B, D
> 448 / 14 = 32, so 32 × 32 = 1,024 patches; 2 × 2 per slot gives 256. masked_scatter fills the slots with the image vectors. The image is input only: −100 is PyTorch’s ignore label.
> *Training Paradigms I · images as tokens*

### Mock B Q9 · scaling laws
<small>from [[GenAI - Mock Test B|Mock B]]</small>

**What is true about scaling laws?**

- [ ] **A** Loss falls as a power law in compute: a straight line on log–log axes
- [ ] **B** When compute grows, compute-optimal parameters and tokens grow by roughly the same factor
- [ ] **C** Chinchilla: compute-optimal training uses about 20 tokens per parameter
- [ ] **D** The Chinchilla-optimal model also minimises the cost of serving it

> [!success]- Answer: A, B, C
> Kaplan: log-linear trends in C, D, N. Chinchilla: N_opt ∝ C^0.49, D_opt ∝ C^0.51. It optimises training compute only: serving a bigger model costs more, which is why the pilot over-trains a small one.
> *Training Paradigms I · scaling laws*

### Mock B Q10 · mixture of experts
<small>from [[GenAI - Mock Test B|Mock B]]</small>

**The pilot activates 1 shared + 8 routed experts out of 256 routed, each 768 wide, with d = 2,048. What is true?**

- [ ] **A** It has about 30B parameters in total but per-token compute like a 2B model
- [ ] **B** The active feed-forward width is 9 · 768 = 6,912, about 3.4 · d
- [ ] **C** The router picks the experts once per document
- [ ] **D** During training every token goes through all 256 experts

> [!success]- Answer: A, B
> Only the selected experts run; the router decides per token (and per layer). Total ÷ active ≈ 32 keeps the knowledge capacity of a much larger model at the compute of the active part.
> *Training Paradigms I · mixture of experts*

### Mock B Q11 · the loss
<small>from [[GenAI - Mock Test B|Mock B]]</small>

**The training loss per token is −log p, with p the probability given to the true next token. What is true?**

- [ ] **A** A model that spreads probability evenly over a 151,936-token vocabulary has a loss of about 12
- [ ] **B** p = 0.5 gives a loss of about 0.69
- [ ] **C** The batch loss is the sum of the GPUs’ mean losses
- [ ] **D** p = 1 gives a loss of 0

> [!success]- Answer: A, B, D
> −log 1 = 0, −log 0.5 = ln 2 ≈ 0.69, −log(1/151,936) = ln 151,936 ≈ 11.9. The batch loss is the mean over all tokens; with equal counts per GPU that is the mean of the GPUs’ means, not their sum.
> *Training Paradigms I · the loss*

### Mock B Q12 · gradient clipping
<small>from [[GenAI - Mock Test B|Mock B]]</small>

**Gradients are clipped to norm 1.0. What is true?**

- [ ] **A** A gradient of norm 0.5 is left as it is
- [ ] **B** A gradient of norm 8 is scaled down to norm 1, keeping its direction
- [ ] **C** Each weight’s gradient is clipped to [−1, 1] on its own
- [ ] **D** It guards against the huge gradient of a loss spike

> [!success]- Answer: A, B, D
> The norm of the gradient of all weights together is computed; if it exceeds 1.0, the whole vector is rescaled. Usual steps are untouched; a spike step keeps its direction but gets length 1.
> *Training Paradigms I · gradient clipping*

### Mock C Q9 · H100 compute
<small>from [[GenAI - Mock Test C|Mock C]]</small>

**An H100 delivers about 989 TFLOP/s in bf16 at peak. What is true?**

- [ ] **A** Model FLOPs utilisation is the share of the training tokens actually used
- [ ] **B** 40,000 H100-hours at 35 % give about 4.9 · 10²⁴ FLOPs
- [ ] **C** At 35 % model FLOPs utilisation, one H100-hour gives about 1.2 · 10¹⁸ useful FLOPs
- [ ] **D** One H100-hour at peak is about 3.56 · 10¹⁸ FLOPs

> [!success]- Answer: C, D
> 989 · 10¹² · 3,600 ≈ 3.56 · 10¹⁸; × 0.35 ≈ 1.23 · 10¹⁸; × 40,000 ≈ 4.9 · 10²², not 10²⁴. MFU is the share of the GPU’s peak spent on useful model maths; the rest goes to communication and memory waits.
> *Training Paradigms I · H100 compute*

### Mock C Q10 · cleaning and decontamination
<small>from [[GenAI - Mock Test C|Mock C]]</small>

**What is true about preparing web data?**

- [ ] **A** Without it, benchmark scores can measure memorisation instead of ability
- [ ] **B** Deduplication removes exact copies only
- [ ] **C** A quality classifier can drop low-quality pages
- [ ] **D** Decontamination removes training pages that contain benchmark test items

> [!success]- Answer: A, C, D
> Filter (heuristics), deduplicate (exact copies, near copies, repeated passages), classify (topic, quality). The GSM8K problem posted on a forum with its answer gets dropped.
> *Training Paradigms I · cleaning and decontamination*

### Mock C Q11 · inputs and targets
<small>from [[GenAI - Mock Test C|Mock C]]</small>

**A training step uses inputs = batch[:, :-1] and targets = batch[:, 1:]. What is true?**

- [ ] **A** A row of 4,097 token IDs gives 4,096 predictions
- [ ] **B** The model sees the target token as input at the same position
- [ ] **C** Positions labelled −100 are still counted in the loss, with weight 1
- [ ] **D** The target at position t is the token at position t + 1 of the row

> [!success]- Answer: A, D
> Inputs and targets are the same row shifted by one. With the causal mask, position t never sees token t + 1. −100 is PyTorch’s ignore label: those positions (e.g. image slots) get no loss.
> *Training Paradigms I · inputs and targets*

### Mock C Q12 · parallelism
<small>from [[GenAI - Mock Test C|Mock C]]</small>

**Training runs on many GPUs. What travels between them?**

- [ ] **A** Data parallel: gradients, averaged once per step
- [ ] **B** Tensor parallel: partial results, summed after every layer
- [ ] **C** Pipeline parallel: nothing, because each GPU holds all layers
- [ ] **D** Expert parallel: tokens, sent to their experts and back

> [!success]- Answer: A, B, D
> Pipeline parallel splits the model by layers: activations travel forward and their gradients back. FSDP / ZeRO-3 gathers each layer’s weights when needed and frees them afterwards.
> *Training Paradigms I · parallelism*

### Extra Q8 · width vs depth
<small>from [[GenAI - Extra Questions|Extra]]</small>

**The pilot has 2B active parameters, d = 2,048, 24 blocks and 16 heads. What is true?**

- [ ] **A** Each head has 128 dimensions
- [ ] **B** d ÷ blocks ≈ 85, inside the good range of 20–250
- [ ] **C** 4,096 × 3 blocks would be too deep
- [ ] **D** Inside that range, loss changes by only about 1 %

> [!success]- Answer: A, B, D
> 2,048 / 24 ≈ 85; 2,048 / 16 = 128. 4,096 × 3 is too wide, 1,024 × 116 too deep (Kaplan Fig. 5, EuroLLM-1.7B’s shape).
> *Training Paradigms I · width vs depth*

### Extra Q9 · tokenizer fertility
<small>from [[GenAI - Extra Questions|Extra]]</small>

**The same text needs 7.2× as many tokens in Greek as in English with GPT-2’s tokenizer. What is true?**

- [ ] **A** Less Greek text fits into a fixed context window
- [ ] **B** Greek text costs more compute per sentence
- [ ] **C** A tokenizer trained on EU languages reduces the factor
- [ ] **D** Training longer removes the problem

> [!success]- Answer: A, B, C
> Fertility is a property of the tokenizer, fixed before training. More tokens per sentence means more compute, higher API cost, shorter effective context and slower generation.
> *Training Paradigms I · tokenizer fertility*

### Extra Q10 · packing
<small>from [[GenAI - Extra Questions|Extra]]</small>

**Documents are packed into training rows of 4,096 tokens. What is true?**

- [ ] **A** A document longer than 4,096 tokens is dropped
- [ ] **B** Packing wastes less compute than padding each document to the row length
- [ ] **C** Every document must be exactly 4,096 tokens long
- [ ] **D** Every document is followed by <|endoftext|>

> [!success]- Answer: B, D
> Join all documents, each ended with <|endoftext|> (151643), and cut into rows: 0 % waste vs 44 % with padding. A long document is cut across rows.
> *Training Paradigms I · packing*

### Extra Q11 · optimiser and schedule
<small>from [[GenAI - Extra Questions|Extra]]</small>

**The update is θ ← θ − ηₜ · Δₜ. What is true?**

- [ ] **A** With AdamW, Δₜ is simply the raw gradient of the current batch
- [ ] **B** In a warmup–stable–decay schedule the learning rate never changes
- [ ] **C** ηₜ comes from the learning-rate schedule, e.g. warmup, plateau, decay
- [ ] **D** AdamW smooths the gradient and sizes the step per weight

> [!success]- Answer: C, D
> Optimiser (Δ): one batch’s gradient is noisy and badly scaled → smooth it (momentum) and size it per weight (AdamW, Muon). Scheduler (η): start gently, go big, end small.
> *Training Paradigms I · optimiser and schedule*

### Extra Q12 · images in the architecture
<small>from [[GenAI - Extra Questions|Extra]]</small>

**Ways to add images to a language model. What is true?**

- [ ] **A** In the adapter route, text and images learn together from the start
- [ ] **B** A native model without an encoder is the most proven route among frontier models
- [ ] **C** Adding images late in training (50 % in the last fifth) scored best in Kimi K2.5’s comparison
- [ ] **D** An adapter joins a separately trained vision encoder to a finished language model through a projector

> [!success]- Answer: D
> Adapter (Qwen2.5-VL): cheap, but images join a model built for text. Native with encoder (Kimi K3): learn together, one big joint run. Native without encoder (Fuyu-8B): simplest, less proven. Early, a little at a time (10 %) was best on images and on text.
> *Training Paradigms I · images in the architecture*

<!-- end of test questions -->

Related: [[GenAI - Glossary|Glossary: symbols & variables]] · [[2026-10-28 - Theoretical Foundations of Language Modelling|Theoretical Foundations of Language Modelling]] · [[2026-11-11 - Training Paradigms II|Training Paradigms II]] · [[GenAI Lab]]
