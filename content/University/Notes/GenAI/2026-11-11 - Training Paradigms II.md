---
publish: true
tags: [genai, lecture, test1]
lecture: 2026-11-11
lecturer: Johannes Oster
site: none
course: Generative AI (194.207)
---

# 04 · Training Paradigms II (11 Nov) – mid-training & post-training

> [!abstract] The story
> - Pre-training produced a base model that continues text.
> - **Mid-training** (step 4) does three things at the end of pre-training: **① cool down the learning rate, ② finish on the best data, ③ extend the context** (4,096 → 1M).
> - **Post-training** (step 5) turns the base model into an assistant, either by learning from **a dataset** (SFT: examples to copy; DPO: pairs, one better) or from **its own output**, scored by a **teacher** (on-policy distillation), a **reward model / judge** (RLHF, RLAIF) or a **checker** (RLVR).
> - The RL methods differ in where the **baseline** comes from: **PPO** (value model) vs **GRPO** (group mean).

> [!info] No 5w1h page for this lecture
> Part of the long-context material (attention cost, KV cache, GQA, MLA, "four ways") is on the [Training Paradigms 5w1h page](https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:context-later), embedded below. For everything else, use the **KV cache**, **DPO / RM**, **KL**, **PPO / GRPO** and **YaRN** tabs of [[GenAI Lab]].

<iframe src="https://5w1h.academy/introduction-to-generative-ai/module-01/training-paradigms/#sec:context-later" width="100%" height="650" style="border:1px solid #8884; border-radius:8px;"></iframe>

---

# Part A · Mid-training

## A1 · Cool down the learning rate

> [!tip] In plain words
> Training is like **hiking down into a valley in fog**. With a big learning rate you take big leaps: great for crossing the landscape quickly, but near the bottom you bounce from one slope to the other. Lowering the learning rate at the end (cool-down) lets you take small steps and **settle at the bottom**, which gives a big drop in loss. That final phase is also the best moment to feed the best data.

With a warmup–stable–decay schedule the loss landscape picture is: **travel** (high LR, big jumps across the landscape) → **bounce** (still high LR, bouncing between the walls of a valley) → **settle** (LR decays → steps get small → slides to the bottom). The decay phase gives a big loss drop and is the natural place to switch data.

> [!example]- Slide 39 (click to open)
> ![[GenAI-L4-slajd39.png|720]]

---

## A2 · Finish on the best data

> [!tip] In plain words
> What a model saw last sticks best, so the last slice of training uses the **best material**: maths, code, reasoning, Q&A, worked solutions written out step by step. The goal is that the model can already **sometimes** solve a problem the right way. Reinforcement learning later can only strengthen behaviour the model already produces now and then; it cannot invent it.

The last, small slice of training uses a higher-quality mix: web, code, **maths**, **reasoning**, **Q&A**, PDFs, including **thinking traces** written as plain text, *without* chat format:

```
17 × 24 = ?
17 × 20 = 340, 17 × 4 = 68
340 + 68 = 408
check: 408 ÷ 24 = 17 ✓
answer: 408
```

Why: the model then already *sometimes* produces correct reasoning, and RL later only has to make the good behaviour more likely ("after mid-training: some ✓ → after RL: mostly ✓"). RL can only reinforce what the model can already sample.

> [!example]- Slide 40 (click to open)
> ![[GenAI-L4-slajd40.png|720]]

---

## A3 · Extend the context – 4,096 → 1,000,000

> [!tip] In plain words
> Reading a million tokens at once has three problems, like asking a student to read a whole library at once: **(1)** attention compares every token with every other one, so the work grows with the **square** of the length; **(2)** the position system (RoPE) has never seen such large positions; **(3)** there are very few really long documents to practise on. Fix (1) in the architecture from the start; fix (2) and (3) at the end of training.

Three obstacles, three fixes:

| Obstacle | Fix | When |
|---|---|---|
| **1 Attention**: $n\times n$ scores per layer (8 tokens → 64 scores; cost and memory grow **quadratically**) | make attention affordable (GQA, MLA, windows, sparse, linear) | **from the start** (architecture) |
| **2 Positions**: RoPE never saw angles beyond the training length | stretch the positions (YaRN) | **in mid-training** |
| **3 Training data**: very few long documents (most docs are ≤ 4K) | train on long rows, at the end | **in mid-training** |

> [!example]- Slide 42 (click to open)
> ![[GenAI-L4-slajd42.png|720]]

### Excursus – attention cost and the KV cache

> [!tip] In plain words
> When a model writes, each new token needs the keys and values of **all earlier tokens**. Recomputing them every time would be absurd, so they are kept in a **notebook**: the KV cache. For each new token you compute only its own query, key and value, look things up in the notebook, and add one new line to it. Old lines never change, because thanks to the mask an old token never looks at newer ones.
> The catch: the notebook grows with every token and can reach hundreds of gigabytes for a long context. GQA (several query heads share one key/value head) and MLA (store a compressed summary per token) make each line smaller.

Training/prefill: $\text{softmax}(QK^\top/\sqrt{d_k})V$ with $Q,K \in \mathbb{R}^{n\times d_k}$ → an **$n\times n$** weight matrix per head and layer.

**Generation with a KV cache.** Generating token $t+1$ needs: the new token's $q$, scored against the $k$ of **every** earlier token, mixing their $v$.

| | What happens for each new token |
|---|---|
| ✓ | **only the new token** goes through all the layers (embedding, q/k/v, attention, FFN, next-token distribution) |
| ✓ | its **query** is compared with the **cached keys** of all earlier tokens (+ its own key) |
| ✓ | its **key and value** are **appended** to the cache for later tokens |
| ✗ | earlier tokens' keys/values are **never updated**: because of the causal mask, a token never sees later tokens, so its $k,v$ never change |

- **Prefill** = reading the prompt: all prompt tokens at once, in parallel (compute-bound) → fills the cache.
- **Decode** = writing: one token per step; each step must read **all weights + the whole cache** from memory → **memory-bound**.

**Cache size per token** (bf16 = 2 bytes):

$$\text{bytes/token}=2\,(\text{K and V})\times L\,(\text{layers})\times h_{kv}\,(\text{KV heads})\times d_{head}\times 2\,(\text{bytes})$$

Pilot with full multi-head attention: $2\cdot24\cdot16\cdot128\cdot2$ = 196,608 B ≈ **192 KiB per token** → at 1M tokens ≈ **200 GB** for one sequence. Hence:

**Grouped-query attention (GQA).** Several query heads **share** one K/V head. Slide: 8 query heads, 2 K/V groups → cache ÷ 4. Quality nearly unchanged. (MQA = all query heads share 1 K/V.)

**Multi-head latent attention (MLA, DeepSeek-V3).** Don't cache K and V; cache a small **latent** per token:

| Symbol | Meaning | DeepSeek-V3 size |
|---|---|---|
| $X$ | token vectors | $d_{model}$ = 7,168 |
| $W^{DKV}$ | **D**own-projection to the joint **K**V latent | 7,168 → 512 |
| $C = XW^{DKV}$ | the cached latent ($c_j$ for token $j$) | $d_c$ = **512** per token |
| $W^{UK}$, $W^{UV}$ | **U**p-projections that rebuild keys / values | 512 → $h\cdot d_{head}$ = 16,384 |
| $K = CW^{UK}$, $V=CW^{UV}$ | full keys and values, recomputed | 16,384 each |

Cache 512 numbers instead of 2 × 16,384 → **64× smaller**. Trick: $W^{UK}$ can be **absorbed into the query**: $q_t k_j^\top = (q_t W^{UK\top})\,c_j^\top$, so K never has to be materialised.

**Four ways to make long attention cheaper** (what each new token looks at):

| Way | Idea | Memory per new token |
|---|---|---|
| full attention | every earlier token, full K/V | grows with $n$, full size |
| **compress each token** | keep every token, but each one smaller (GQA, MLA) | grows with $n$, smaller |
| **look nearby** | sliding window: only the last $w$ tokens (usually mixed with some full-attention layers; Olmo 3 does this) | fixed: $w$ tokens |
| **squeeze, then select** | cheap look at compressed blocks, then full attention only to the few selected ones (sparse attention) | grows, but only a few read in full |
| **a fixed-size state** | linear attention / recurrent state that summarises everything so far | constant |

(Slide sources: MiniMax-M2, DeepSeek, Olmo 3, DeepSeek-V4, Kimi K3. Frontier models mix several of these across layers.)

**Try it: KV cache size: GQA and MLA**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=kvcache" width="100%" height="640" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slides 43, 44, 45, 46, 47 (click to open)
> ![[GenAI-L4-slajd43.png|720]]
>
> ![[GenAI-L4-slajd44.png|720]]
>
> ![[GenAI-L4-slajd45.png|720]]
>
> ![[GenAI-L4-slajd46.png|720]]
>
> ![[GenAI-L4-slajd47.png|720]]

### Stretching the positions – YaRN

> [!tip] In plain words
> RoPE's clock hands: fast hands went round many times during training, so every angle is familiar. Slow hands barely moved: at position 50,000 they would point to angles **never seen in training**, and the model gets confused. YaRN **slows down only the slow hands** (by the scale factor) so that the long text maps onto angles the model knows, and leaves the fast hands alone, because they carry the fine word order nearby. Slowing all hands would blur that local order.

RoPE pair $i$ turns by $\theta_i=b^{-2(i-1)/d}$ per position. Olmo 3 7B ($b$ = 500,000, head dim 128 → 64 pairs), trained on 8K rows:

| Pair | Turns within 0–8K | Status during training | YaRN (8K → 64K, scale 8) |
|---|---|---|---|
| pair 1 (fastest) | 1,304 full turns | every angle seen many times | **unchanged** |
| pair 16 | 60 turns | every angle seen | unchanged |
| pair 24 | 12 turns | | ÷2.3 (partly) |
| pair 30 | 3.4 turns | | ÷5.2 (partly) |
| pair 48 (slow) | only **31°** | most angles **never seen** | **÷8** |

**Why:** at position 50K a slow pair would reach angles the model never saw → garbage. Dividing its speed by the scale factor 8 maps 0–64K into the angle range seen in 0–8K (**interpolation**). Fast pairs are left alone (they carry fine **local** order and have seen all angles anyway). Rule (YaRN default thresholds): pairs with > 32 turns in the training length: untouched; < 1 turn: ÷ scale; in between: blended. Olmo 3: 19 fast pairs untouched, 16 partly, 29 slow ÷8. (Plain *position interpolation* would divide **all** pairs by 8 → blurs nearby order.)

**Try it: YaRN: which pairs get slowed down**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=yarn" width="100%" height="760" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slide 48 (click to open)
> ![[GenAI-L4-slajd48.png|720]]

### Long rows, at the end (Olmo 3)

> [!tip] In plain words
> To learn long reading you need long practice texts in one piece. So the last stage uses long rows (64K) with real long documents. Short documents fill the rest of each row, and a mask makes sure each document only "sees" itself, like **putting dividers between unrelated papers in one folder**. Two thirds of the mix stays short data so the model does not forget its short-text skills.

- Pre- and mid-training: rows of **8K**; long documents are cut across rows, so the model never sees one in one piece.
- Long-context stage: rows of **64K**; a 30K PDF fits in one piece for the first time; short docs fill the rest.
- Stage: **50B tokens**: **34%** long documents (8K–64K: PDFs, synthetic tasks) + **66%** the short mid-training mix "so short skills hold".
- **Intra-document masking**: in a packed row, put $-\infty$ **between documents** so each token only sees its own document (the mask becomes block-diagonal-causal). Unrelated docs in the same row would otherwise leak into each other.

Extension factors reported: Olmo 3 7B ×8 (1 step), DeepSeek-V3 ×32 (2 steps), Llama 3 ×16 (6 stages), GLM-5 ×6 (2 steps), DeepSeek-V4 ×256 to 1M **in pre-training**. 1M is today's frontier, reached with new attention.

> [!example]- Slides 49, 50 (click to open)
> ![[GenAI-L4-slajd49.png|720]]
>
> ![[GenAI-L4-slajd50.png|720]]

---

# Part B · Post-training

> [!tip] In plain words
> After pre-training the model can **continue text**, but it is not yet an assistant. Post-training teaches it how to behave. The big question for every method: **where does the feedback come from?** Either from a fixed dataset, or from grading the model's own answers (by a teacher, a reward model, or a program).

**Where the learning signal comes from** (same tree as slide 74 below):

| The model learns from … | What it holds / who scores | Method |
|---|---|---|
| **a dataset** | examples to copy (written by people, a stronger model, or the best of its own tries) | **SFT** |
| **a dataset** | pairs, one answer better than the other | **DPO** |
| **its own output** | a **teacher** grades every token | **on-policy distillation** |
| **its own output** | a **reward model or judge** gives one score | **RLHF / RLAIF** |
| **its own output** | a **checker** (program) says right or wrong | **RLVR** |

For the last two the update needs a **baseline**: PPO takes it from a value model, GRPO from the group mean.

> [!example]- Slide 74 (click to open)
> ![[GenAI-L4-slajd74.png|720]]

---

## B1 · SFT – examples to copy

> [!tip] In plain words
> Supervised fine-tuning is **learning by imitation**: show good (question, answer) pairs and train with the normal next-token loss. It is fast and simple, but the model only ever sees good answers; it never learns what a bad answer looks like or why it is bad.

Ordinary next-token loss, on (prompt, answer) pairs. Where the answers come from:
- **curated dataset**: people write them,
- **distillation**: a stronger model writes them,
- **rejection sampling**: the policy writes several, the best are kept.

The model only learns to **imitate**; it never sees what a *bad* answer looks like.

---

## B2 · DPO – pairs, one better

> [!tip] In plain words
> DPO learns from **pairs**: for one question, answer A was preferred to answer B. The model is pushed to raise A and lower B, **measured relative to a frozen copy of itself** (the reference). Why relative? Absolute probabilities depend on length and wording; what matters is whether training moved A up compared to where it started, and B down.
> Think of a **taste test**: you do not need to know how good each dish is in points, only which of the two people preferred. No reward model, no sampling during training: just a fixed list of pairs.

**Where "chosen" comes from:** people compare (InstructGPT: rank 4–9 answers) · a model judges (Tülu 3: GPT-4o rates) · bigger model's answer wins (Olmo 3: Qwen3-32B's answer is "chosen" over Qwen3-0.6B's). **Labelled once: a fixed dataset** (offline, no sampling during training).

$$\mathcal{L}_{DPO}=-\log\sigma\Big(\beta\big[\underbrace{(\log p_{policy}(y^+)-\log p_{ref}(y^+))}_{\text{gain}^+}-\underbrace{(\log p_{policy}(y^-)-\log p_{ref}(y^-))}_{\text{gain}^-}\big]\Big)$$

| Variable | Meaning |
|---|---|
| $x$ | the prompt ("Why do cows moo?") |
| $y^+, y^-$ | chosen / rejected answer |
| $\log p(y)$ | **sum** of the token log-probs of the whole answer |
| $p_{policy}$ | model being trained |
| $p_{ref}$ | **frozen copy of the SFT model** |
| gain | how much more likely the policy makes an answer *than the reference does* |
| $\beta$ | 0.1. Plays the role of the KL weight from RLHF: **larger β → policy stays closer to the reference** (the loss is already satisfied by smaller log-ratio changes) |
| $\sigma$ | sigmoid $1/(1+e^{-x})$ |

**Worked example (slide):** chosen: policy −38, ref −40 → gain⁺ = +2; rejected: policy −37, ref −35 → gain⁻ = −2.
$\mathcal{L}=-\log\sigma(0.1\cdot[2-(-2)])=-\log\sigma(0.4)=-\log0.60=\mathbf{0.51}$.
Note the policy actually gives the *rejected* answer a higher absolute log-prob (−37 > −38): DPO cares about the change **relative to the reference**, not absolute likelihoods. Minimising pushes gain⁺ up and gain⁻ down.

DPO needs **no reward model and no sampling**: only policy + frozen reference + a fixed preference dataset.

**Try it: DPO loss**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=dpo" width="100%" height="520" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slide 58 (click to open)
> ![[GenAI-L4-slajd58.png|720]]

---

## B3 · On-policy distillation – a teacher, token by token

> [!tip] In plain words
> The student writes its own answer; a stronger teacher **reads it once** and says, for every single token, how likely the teacher itself would have written it. Tokens the teacher likes more than the student → push up; tokens the teacher strongly disagrees with (like "418" for 17 × 24) → push down.
> It is like a **driving instructor commenting on every manoeuvre of your own drive**, rather than you watching videos of perfect drivers (SFT). You learn from your own mistakes, with feedback at every step.

The **student (policy)** writes its own answer ("17 × 24 = 418"). The **teacher** does **one forward pass** over the student's text and gives its probability for each of the student's tokens.

$$\text{grade}_t=\log p_{teacher}(\text{token}_t)-\log p_{student}(\text{token}_t),\qquad \mathcal{L}=-\sum_t \text{grade}_t\cdot\log p_{student}(\text{token}_t)$$

| token | $p_{student}$ | $p_{teacher}$ | grade |
|---|---|---|---|
| 17 × 24 | 0.60 | 0.70 | +0.15 (teacher likes it more → push up) |
| = | 0.90 | 0.90 | 0 |
| 418 | 0.30 | 0.02 | **−2.71** (teacher strongly disagrees → push down) |

"Grade held fixed": the grade is treated as a constant **reward per token** (no gradient flows through it). Advantages: on-policy (learns on its own mistakes, unlike SFT/offline distillation), and **dense** feedback for every token (unlike a single reward at the end).

**Try it: On-policy distillation grades**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=distill" width="100%" height="440" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slide 60 (click to open)
> ![[GenAI-L4-slajd60.png|720]]

---

## B4 · RLHF – a reward model

> [!tip] In plain words
> Step 1: people rank answers; a **reward model** learns to imitate their taste (only score differences matter: "A beats B"). Step 2: the model writes answers, the reward model scores them, and the model is pushed towards higher scores.
> The danger is **gaming the judge** (Goodhart's law): the reward model is only an imitation of people, and if the policy optimises it too hard it finds tricks the reward model likes but people don't. Like a student who learns the examiner's quirks instead of the subject. The **KL leash** pulls the policy back towards the original model, more strongly the further it drifts.

**① Train the reward model** (Bradley–Terry). People rank several answers (e.g. 4 → $\binom42=6$ pairs: 1>2, 1>3, 1>4, 2>3, 2>4, 3>4). For each pair:

$$\mathcal{L}_{RM}=-\log\sigma(r^+-r^-)$$

Example: $r^+=1.4$, $r^-=-0.6$ → $-\log\sigma(2.0)=\mathbf{0.13}$. Update the **reward model's** weights. Only *differences* of scores matter.

**② RL with the reward model.** Policy writes a fresh answer → reward model scores it (0.8) → update. Simplest loss ("REINFORCE"):

$$\mathcal{L}=-r\cdot\sum_t\log p_{policy}(\text{token}_t)$$

= SFT on its **own** answer, every token weighted by the **one** reward.

**③ Over-optimisation and the KL leash.** The reward model is only a proxy. As the policy drifts from where it started, the **reward model's score keeps rising** while the **true quality first rises, then falls** (Goodhart; Gao et al.). Fix: a leash that pulls back harder the further the policy drifts:

$$\text{KL}=\sum_{\text{token}}p_{policy}\cdot(\log p_{policy}-\log p_{ref})$$

Example over the next-word distribution after "Cows moo to …": ref (SFT) = call 0.50 / warn 0.30 / eat 0.20; policy = 0.80 / 0.15 / 0.05.
$\text{KL}=0.80(+0.47)+0.15(-0.69)+0.05(-1.39)=\mathbf{0.20}$. KL = 0 iff the distributions are identical; always ≥ 0; not symmetric.

$$r_{\text{used}}=r_{RM}-\beta\cdot\text{KL}=0.8-0.1\cdot1.5=\mathbf{0.65}$$

The **reference is frozen** (the SFT model); the **reward model is frozen** during RL; only the policy changes.

**RLAIF:** the labels come from an AI judge guided by a **constitution written by people** (< \$0.01 per label). Rubric per prompt (gives a reason ✓, names calves/herd ✓, mentions other reasons ✗, short ✓) → 0.75 vs 0.25.

**Try it: Reward model loss**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=rm" width="100%" height="470" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

**Try it: KL and the KL-penalised reward**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=kl" width="100%" height="560" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slides 63, 67, 65, 66 (click to open)
> ![[GenAI-L4-slajd63.png|720]]
>
> ![[GenAI-L4-slajd67.png|720]]
>
> ![[GenAI-L4-slajd65.png|720]]
>
> ![[GenAI-L4-slajd66.png|720]]

---

## B5 · RLVR – a checker

> [!tip] In plain words
> When the answer can be **checked by a program**, you do not need a judge at all: the maths answer is 408 or it is not, the code passes its tests or it does not. Reward 1 or 0, no reward model to fool. It only works where correctness is checkable; a poem or comforting a friend has no unit test.

Reward = **1 / 0 from a program**, no reward model: maths (final answer = 408?), code (unit tests in a sandbox), instructions (constraints: under 50 words, no commas – a script checks), puzzles (generated, answer known), format (valid JSON, right tool call).
Works when correctness is **verifiable**. Not for: poems, comforting a friend, summaries, "is this joke funny?" → those need RLHF/RLAIF.

---

## B6 · Why not just the reward? The baseline

> [!tip] In plain words
> If all rewards are positive, "push every answer up in proportion to its reward" makes even the worst answer more likely, just less so. Subtract a **baseline** (what you expected) and the signal becomes "better or worse than expected?": good answers go up, bad answers go down. Like grading **relative to the class average** instead of in absolute points.

With $\mathcal{L}=-r\sum_t\log p$ and rewards 0.8 / 0.5 / 0.3 / 0.1 for four answers, **every** answer is pushed **up** (all $r>0$), the bad ones just less. Subtract a baseline $b$:

$$\mathcal{L}=-(r-b)\cdot\sum_t\log p(\text{token}_t)$$

With $b$ = mean = 0.425: +0.375, +0.075, −0.125, −0.325 → better-than-average pushed up, worse pushed **down**; lower variance. *Where does $b$ come from?* → PPO or GRPO.

> [!example]- Slide 73 (click to open)
> ![[GenAI-L4-slajd73.png|720]]

---

## B7 · PPO

> [!tip] In plain words
> PPO gets its baseline from a **value model** that predicts, at every token, how much reward is still to come. Advantage = what actually came − what was expected, so each token gets its own credit.
> The ratio and clip are a **speed limit**: one batch of answers is reused for several updates, and once a token's probability has changed by more than ~20 % since the answer was written, PPO stops pushing it further. This keeps any single batch from changing the model too much.

$$\mathcal{L}=-\sum_t\min\big(\rho_tA_t,\ \text{clip}(\rho_t,1-\varepsilon,1+\varepsilon)A_t\big)$$

| Symbol | Definition | "Cows moo to call their calves" |
|---|---|---|
| $\text{KL}_t$ | $\log p_{policy}(\text{token}_t)-\log p_{ref}(\text{token}_t)$ (per token) | |
| $-0.1\,\text{KL}_t$ | per-token leash | −0.01, 0.00, −0.02, −0.01, −0.04, −0.07 |
| $r_{RM}$ | reward-model score, **last token only** | 0 … 0, 0.80 |
| $r_t$ | $-0.1\,\text{KL}_t+r_{RM}$ | −0.01, 0.00, −0.02, −0.01, −0.04, **0.73** |
| $G_t$ | return $r_t+r_{t+1}+\dots+r_{last}$ = "what came" after $t$ | 0.65, 0.66, 0.66, 0.68, 0.69, 0.73 |
| $V_t$ | **value model**'s prediction of $G_t$ from tokens ≤ t; trained by $(V_t-G_t)^2$ | 0.40, 0.42, 0.45, 0.50, 0.55, 0.60 |
| $A_t$ | advantage $G_t-V_t$ ($V_t$ = the baseline $b$) | +0.25, +0.24, +0.21, +0.18, +0.14, +0.13 |
| $\rho_t$ | $p_{now}(\text{token}_t)/p_{\text{when written}}(\text{token}_t)$: the same answer is re-used for several updates | 0.45/0.45 = 1.00 → 0.50/0.45 = 1.11 → 0.55/0.45 = **1.22** |
| $\varepsilon$ | clip range, often 0.2 → ratio limited to [0.8, 1.2] | update 3 is past 1+ε: **stop** pushing |

Why the ratio and clip: we reuse one batch of samples for several gradient steps; once a token's probability has moved > 20% from when it was sampled, the clipped term stops giving gradient → no single update moves the policy too far. The $\min$ makes it pessimistic (only clips in the direction that would help).

**Try it: PPO ratio, clip and returns**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=ppo" width="100%" height="780" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slide 75 (click to open)
> ![[GenAI-L4-slajd75.png|720]]

---

## B8 · GRPO (DeepSeekMath)

> [!tip] In plain words
> GRPO drops the value model. Instead it asks the same question **several times** and compares the answers with each other: advantage = how much better than the group average, in units of the group's spread. Every token of an answer gets the same advantage. Like grading a set of attempts **against each other** instead of against a predicted score. If all attempts get the same reward, there is nothing to compare: no learning from that question.

Sample **$G$ answers to the same prompt**, score each ($r_i$), and use the **group** as the baseline:

$$A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)},\qquad \mathcal{L}=-\sum_{i,t}\Big[\min\big(\rho_{i,t}A_i,\ \text{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i\big)-\beta\,\text{KL}_t\Big]$$

Example: $r$ = 0.8, 0.5, 0.3, 0.1 → mean 0.425, std 0.259 → $A$ = **+1.45, +0.29, −0.48, −1.25**. **Every token of answer $i$ gets the same $A_i$**. $\beta=0.04$, KL now **in the loss**.

**Try it: GRPO group advantages**

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=grpo" width="100%" height="520" loading="lazy" style="border:1px solid #8884; border-radius:8px;"></iframe>

> [!example]- Slide 77 (click to open)
> ![[GenAI-L4-slajd77.png|720]]

### PPO vs GRPO

| | PPO | GRPO |
|---|---|---|
| baseline $b$ | $V_t$, a **value model** | mean($r$) of the group |
| advantage | $A_t=G_t-V_t$ | $A_i=(r_i-\text{mean})/\text{std}$ |
| credit | **per token** (varies along the answer) | **same for all tokens** of an answer |
| answers per prompt | 1 | $G$ |
| models in memory | **4**: policy, reference, reward model, value model | **3**: policy, reference, reward model/checker (no value model) |
| KL leash | in the **reward**: $r_t=r_{RM}-\beta\text{KL}_t$ | in the **loss**: $\mathcal{L}=\dots-\beta\text{KL}_t$ |
| ratio, clip | $\rho$, clip($\rho,1-\varepsilon,1+\varepsilon$) | the same |

> [!warning] GRPO edge case
> If all $G$ answers get the same reward (all right or all wrong), std = 0 and every $A_i=0$ → **no learning signal** from that prompt. Prompts must be neither too easy nor too hard.

> [!example]- Slide 78 (click to open)
> ![[GenAI-L4-slajd78.png|720]]

---

## Practice questions (8)

> [!info] Same format as the test
> 0 to 4 options can be correct: decide each option on its own, true or false. Next to each option is the section above that answers it.
> Full timed tests: [[GenAI - Mock Test A|Mock A]] · [[GenAI - Mock Test B|Mock B]] · [[GenAI - Mock Test C|Mock C]] · [[GenAI - Mock Test D|Mock D]] (their questions are not repeated here).

### P1 · the KV cache: prefill and decode
<small>from the official example sheet</small>

**A model generates text one token at a time, with a standard KV cache. What happens for each new token?**

- [ ] **A** Only the new token goes through all the layers <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **B** Its query is compared with the cached keys of all earlier tokens <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **C** Its key and value are stored in the cache for later tokens <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **D** The keys of earlier tokens are updated to include the new token <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Earlier tokens are needed only inside attention, through their keys and values. Because of the mask, a token never sees later tokens, so its key and value never change.

### P2 · RLHF: over-optimization, KL, the loop
<small>from the official example sheet</small>

**Why does RLHF penalize the policy with β · KL to the reference model?**

- [ ] **A** It stops the policy from optimizing the reward model’s score too far <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **B** It keeps the policy close to the SFT model it started from <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **C** It keeps the reward model close to its weights from before RL <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **D** It rewards the policy for tokens the reference model finds unlikely <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> If the policy chases the reward model too far, its score keeps rising while the real quality goes up and then down. The KL term pulls the policy back towards the frozen SFT model, harder the further it moves. The reward model does not change during RL.

### P3 · cool-down

**Mid-training with a warmup–stable–decay schedule. What is true?**

- [ ] **A** The decay phase gives a big drop in loss <small>· look in [[2026-11-11 - Training Paradigms II#A1 · Cool down the learning rate|TP II §A1 Cool down the learning rate]]</small>
- [ ] **B** The decay phase is a natural place to switch to the best data <small>· look in [[2026-11-11 - Training Paradigms II#A1 · Cool down the learning rate|TP II §A1 Cool down the learning rate]]</small>
- [ ] **C** The learning rate is highest at the end of training <small>· look in [[2026-11-11 - Training Paradigms II#A1 · Cool down the learning rate|TP II §A1 Cool down the learning rate]]</small>
- [ ] **D** “Bounce”: the learning rate is still high and the steps jump between the walls of a valley <small>· look in [[2026-11-11 - Training Paradigms II#A1 · Cool down the learning rate|TP II §A1 Cool down the learning rate]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Travel → bounce → settle: once the learning rate decays, the steps get small and the model slides to the bottom of the valley.

### P4 · three obstacles to long context

**Going from 4,096 to 1M tokens of context. What is true?**

- [ ] **A** The attention cost grows quadratically with the number of tokens <small>· look in [[2026-11-11 - Training Paradigms II#A3 · Extend the context – 4,096 → 1,000,000|TP II §A3 Extend the context]]</small>
- [ ] **B** Positions beyond the training length are handled by stretching RoPE (YaRN) in mid-training <small>· look in [[2026-11-11 - Training Paradigms II#A3 · Extend the context – 4,096 → 1,000,000|TP II §A3 Extend the context]]</small>
- [ ] **C** The lack of long documents is fixed in the architecture, from the start <small>· look in [[2026-11-11 - Training Paradigms II#A3 · Extend the context – 4,096 → 1,000,000|TP II §A3 Extend the context]]</small>
- [ ] **D** Olmo 3’s long-context stage has 50B tokens, about a third of them long documents <small>· look in [[2026-11-11 - Training Paradigms II#Long rows, at the end (Olmo 3)|TP II · Long rows, at the end]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Attention: architecture, from the start. Positions and long data: mid-training. Olmo 3: 34 % long documents (8K–64K) + 66 % short mid-training mix.

### P5 · SFT

**What is true about supervised fine-tuning (SFT)?**

- [ ] **A** It uses the ordinary next-token loss on (prompt, answer) pairs <small>· look in [[2026-11-11 - Training Paradigms II#B1 · SFT – examples to copy|TP II §B1 SFT]]</small>
- [ ] **B** In rejection sampling the policy writes several answers and the best are kept <small>· look in [[2026-11-11 - Training Paradigms II#B1 · SFT – examples to copy|TP II §B1 SFT]]</small>
- [ ] **C** It shows the model what a bad answer looks like <small>· look in [[2026-11-11 - Training Paradigms II#B1 · SFT – examples to copy|TP II §B1 SFT]]</small>
- [ ] **D** In distillation, a weaker model writes the answers <small>· look in [[2026-11-11 - Training Paradigms II#B1 · SFT – examples to copy|TP II §B1 SFT]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> SFT only imitates. Answers come from people (curated), a stronger model (distillation) or the policy itself, filtered (rejection sampling).

### P6 · four ways to cheaper attention

**What does each new token look at in the cheaper attention variants?**

- [ ] **A** Sliding window: only the last w tokens, so memory per new token is fixed <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **B** Linear attention: a fixed-size state that summarises everything so far <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **C** GQA and MLA: fewer earlier tokens, each kept at full size <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **D** Sparse attention: a cheap look at compressed blocks, then full attention to a few selected ones <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> GQA and MLA keep every token but make each one smaller in the cache. Frontier models mix several of these across layers.

### P7 · properties of KL

**KL = Σ p_policy (log p_policy − log p_ref). What is true?**

- [ ] **A** It is 0 exactly when the two distributions are identical <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **B** It is never negative <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **C** KL(policy ‖ ref) always equals KL(ref ‖ policy) <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **D** For policy (0.80, 0.15, 0.05) vs reference (0.50, 0.30, 0.20) it is about 0.20 <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> 0.80 · 0.47 + 0.15 · (−0.69) + 0.05 · (−1.39) ≈ 0.20. KL is not symmetric.

### P8 · PPO’s value model

**In PPO for “Cows moo to call their calves”, what is true?**

- [ ] **A** The return Gₜ adds up the rewards from token t to the end <small>· look in [[2026-11-11 - Training Paradigms II#B7 · PPO|TP II §B7 PPO]]</small>
- [ ] **B** The value model is trained with (Vₜ − Gₜ)² <small>· look in [[2026-11-11 - Training Paradigms II#B7 · PPO|TP II §B7 PPO]]</small>
- [ ] **C** Every token of the answer gets the same advantage <small>· look in [[2026-11-11 - Training Paradigms II#PPO vs GRPO|TP II · PPO vs GRPO]]</small>
- [ ] **D** PPO keeps four models in memory: policy, reference, reward model, value model <small>· look in [[2026-11-11 - Training Paradigms II#PPO vs GRPO|TP II · PPO vs GRPO]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Aₜ = Gₜ − Vₜ varies along the answer (+0.25 … +0.13 in the example). Same advantage for all tokens is GRPO.

---

## Exam traps from this lecture

- KV cache: new token only; q vs cached keys; k,v appended; old entries never change (causal mask).
- Decode is **memory-bound**; prefill is parallel.
- GQA shares K/V across query heads; MLA caches a latent and up-projects.
- YaRN slows **slow** RoPE pairs, leaves fast ones; intra-document masking = $-\infty$ between packed docs.
- DPO: offline, no reward model, reference = frozen SFT model, β = 0.1, gains relative to reference.
- RM loss $-\log\sigma(r^+-r^-)$; RM is frozen during RL.
- KL leash: stops over-optimising the reward model, keeps the policy near the SFT model. It does **not** reward unlikely tokens and does **not** constrain the reward model.
- RLVR needs a checkable answer.
- Baseline: lets bad answers get pushed down; reduces variance; does not change what is optimal.
- PPO = value model + per-token credit; GRPO = group mean/std, same advantage for all tokens, no value model, KL in the loss.

Related: [[GenAI - Glossary|Glossary: symbols & variables]] · [[2026-11-04 - Training Paradigms I|Training Paradigms I]] · [[2026-11-18 - Applied Generative AI I|Applied Generative AI I]] · [[GenAI Lab]]
