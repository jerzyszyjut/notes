---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Mock Test C

20 questions in the format of the real test, 40 minutes. The questions of Mock Tests A–D are **not** in the lecture notes, so each test stays fresh.

> [!info] How the test is scored
> 20 questions · 40 minutes · closed book · **0 to 4 options may be correct**, and you are not told how many. Treat every option as its own true/false question.
> Per question (1.5 points): **+1.5 ÷ n** per correct tick (n = number of correct options), **−0.75** per wrong tick, never below 0. If no option is correct, **no tick = 1.5**.
> Next to every option: the section of the notes where the answer is.

---

## Q1 · the probabilistic view
<small>Intro</small>

**What is true about the probabilistic view of AI?**

- [ ] **A** The goal of a generative model is p_θ(x) ≈ p(x) <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>
- [ ] **B** Qwen3-0.6B-Base finds “The cow says moo” about four times as likely as “The cow says meow” <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>
- [ ] **C** A discriminative model can generate new pictures by sampling p(cow | picture) <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>
- [ ] **D** A discriminative model estimates p(picture) <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Regularities of the world show up as differences in probability. Discriminative: p(cow | picture), a score for a given input, nothing to sample pictures from. Generative: p(picture), the data itself.

---

## Q2 · BPE tokenization
<small>Intro</small>

**A byte-level BPE vocabulary learned 256 ·c, 258 the, 260 ow, 262 ·the, 263 ·cow, 271 ·say, 272 ·café, 273 ·in. It tokenizes “the cows in the café say wow” as (258, 263, 115, 273, 262, 272, 271, 32, 119, 260). What is true?**

- [ ] **A** 32 is the byte for a space, kept as its own token here <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>
- [ ] **B** <|endoftext|> was learned as a merge of frequent bytes <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>
- [ ] **C** “·cows” is split into ·cow and s because no merge produced ·cows <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>
- [ ] **D** “wow” is split because no merge covers ·wow as a whole <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> Tokenizing applies the learned merges; whatever is not covered stays as smaller pieces, down to single bytes (115 = s, 119 = w, 32 = space). Special tokens are added after the merges, not learned from pairs.

---

## Q3 · symbolic vs subsymbolic
<small>Intro</small>

**What is true about symbolic and subsymbolic AI?**

- [ ] **A** Symbolic AI learns its rules from data <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **B** A neural network is interpretable because each weight stands for a concept <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **C** Subsymbolic systems are corrected by editing their rules <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **D** A random forest is symbolic, because its trees are made of rules <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>

> [!success]- Answer (click to open)
> **none** · A ✗ · B ✗ · C ✗ · D ✗
>
> None. A random forest learns from data: subsymbolic. Subsymbolic models are corrected only by retraining; symbolic rules are written by people; neural network weights are hard to interpret. (No tick at all scores 1.5.)

---

## Q4 · which parts mix tokens
<small>LM</small>

**In a decoder-only transformer, which parts let a token’s vector take in information from other positions?**

- [ ] **A** The feed-forward layer <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **B** LayerNorm <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **C** The token embedding <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#1 · Embedding|LM §1 Embedding]]</small>
- [ ] **D** Masked multi-head attention <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#2 · Attention – the core|LM §2 Attention]]</small>

> [!success]- Answer (click to open)
> **D** · A ✗ · B ✗ · C ✗ · D ✓
>
> Only attention compares and mixes tokens. Embedding (a lookup), feed-forward (each token alone) and LayerNorm (over one token’s own numbers) all work per token.

---

## Q5 · dₖ and saturation
<small>LM</small>

**Two heads skip the √dₖ scaling. Head A has dₖ = 16, head B has dₖ = 1,024; q and k have entries of variance 1. What is true?**

- [ ] **A** Head B’s scores typically spread about 8 times wider than head A’s <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **B** Head B’s weights no longer add up to 1 per row <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **C** Less gradient flows back through head B’s softmax <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **D** Head B’s softmax is more likely to give one token nearly all the weight <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> Standard deviation √dₖ: √1,024 / √16 = 32 / 4 = 8. Wide scores saturate the softmax: nearly one-hot, gradients ≈ 0. The softmax still sums to 1 in every case.

---

## Q6 · the causal mask
<small>LM</small>

**A causal mask M is added to the scaled scores before the softmax. What is true?**

- [ ] **A** The diagonal is masked too, so a token cannot attend to itself <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>
- [ ] **B** The mask is applied to V rather than to the scores <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **C** M holds −∞ strictly above the diagonal and 0 elsewhere <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **D** After the softmax, the first row is (1, 0, 0, …) <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>

> [!success]- Answer (click to open)
> **C, D** · A ✗ · B ✗ · C ✓ · D ✓
>
> Each token may see the tokens before it and itself. Row 1 has only itself, so all its weight goes there. The mask acts on the scores; e^(−∞) = 0 gives exactly zero weight to the future.

---

## Q7 · Add & Norm
<small>LM</small>

**A block computes x + Attention(Norm(x)), then x + FeedForward(Norm(x)). What is true?**

- [ ] **A** Attention adds what it found to the token’s vector, which is carried straight up <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **B** The norm is computed across all tokens of the sequence <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **C** The residual path helps gradients flow through deep stacks <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **D** Qwen3’s RMSNorm skips subtracting the mean <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> This is the pre-norm residual stream. The norm works on each token’s own vector. RMSNorm only divides by the root mean square (no μ, no β).

---

## Q8 · RoPE frequencies
<small>LM</small>

**RoPE turns pair i by θᵢ = b^(−2(i−1)/d) per position. What is true?**

- [ ] **A** b sets the maximum number of tokens the model can read <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>
- [ ] **B** Qwen3 uses b = 1,000,000; the original value is 10,000 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>
- [ ] **C** A larger base b makes the slowest pairs turn even more slowly <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>
- [ ] **D** Pair 1 turns fastest <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>

> [!success]- Answer (click to open)
> **B, C, D** · A ✗ · B ✓ · C ✓ · D ✓
>
> θ₁ = b⁰ = 1 rad per position; later pairs are exponentially slower, more so for larger b. The base is not a hard length limit; context length depends on what was trained (and stretching like YaRN).

---

## Q9 · H100 compute
<small>TP I</small>

**An H100 delivers about 989 TFLOP/s in bf16 at peak. What is true?**

- [ ] **A** Model FLOPs utilisation is the share of the training tokens actually used <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>
- [ ] **B** 40,000 H100-hours at 35 % give about 4.9 · 10²⁴ FLOPs <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>
- [ ] **C** At 35 % model FLOPs utilisation, one H100-hour gives about 1.2 · 10¹⁸ useful FLOPs <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>
- [ ] **D** One H100-hour at peak is about 3.56 · 10¹⁸ FLOPs <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>

> [!success]- Answer (click to open)
> **C, D** · A ✗ · B ✗ · C ✓ · D ✓
>
> 989 · 10¹² · 3,600 ≈ 3.56 · 10¹⁸; × 0.35 ≈ 1.23 · 10¹⁸; × 40,000 ≈ 4.9 · 10²², not 10²⁴. MFU is the share of the GPU’s peak spent on useful model maths; the rest goes to communication and memory waits.

---

## Q10 · cleaning and decontamination
<small>TP I</small>

**What is true about preparing web data?**

- [ ] **A** Without it, benchmark scores can measure memorisation instead of ability <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **B** Deduplication removes exact copies only <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **C** A quality classifier can drop low-quality pages <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **D** Decontamination removes training pages that contain benchmark test items <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> Filter (heuristics), deduplicate (exact copies, near copies, repeated passages), classify (topic, quality). The GSM8K problem posted on a forum with its answer gets dropped.

---

## Q11 · inputs and targets
<small>TP I</small>

**A training step uses inputs = batch[:, :-1] and targets = batch[:, 1:]. What is true?**

- [ ] **A** A row of 4,097 token IDs gives 4,096 predictions <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **B** The model sees the target token as input at the same position <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **C** Positions labelled −100 are still counted in the loss, with weight 1 <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **D** The target at position t is the token at position t + 1 of the row <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> Inputs and targets are the same row shifted by one. With the causal mask, position t never sees token t + 1. −100 is PyTorch’s ignore label: those positions (e.g. image slots) get no loss.

---

## Q12 · parallelism
<small>TP I</small>

**Training runs on many GPUs. What travels between them?**

- [ ] **A** Data parallel: gradients, averaged once per step <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>
- [ ] **B** Tensor parallel: partial results, summed after every layer <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>
- [ ] **C** Pipeline parallel: nothing, because each GPU holds all layers <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>
- [ ] **D** Expert parallel: tokens, sent to their experts and back <small>· look in [[2026-11-04 - Training Paradigms I#Parallelism – what travels between GPUs|TP I · Parallelism]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Pipeline parallel splits the model by layers: activations travel forward and their gradients back. FSDP / ZeRO-3 gathers each layer’s weights when needed and frees them afterwards.

---

## Q13 · MLA and GQA
<small>TP II</small>

**What is true about grouped-query attention (GQA) and multi-head latent attention (MLA)?**

- [ ] **A** Both make the KV cache smaller <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **B** In GQA several query heads share one key/value head <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **C** MLA caches a small latent per token (512 numbers in DeepSeek-V3) instead of full keys and values <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **D** In MLA, W^UK can be folded into the query, so the full keys need not be built <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>

> [!success]- Answer (click to open)
> **A, B, C, D** · A ✓ · B ✓ · C ✓ · D ✓
>
> MLA: C = X W^DKV (7,168 → 512); K = C W^UK and V = C W^UV (512 → 16,384). qₜkⱼᵀ = (qₜ W^UKᵀ) cⱼᵀ. 512 vs 2 · 16,384: 64× less cache.

---

## Q14 · packing long rows
<small>TP II</small>

**In the long-context stage, a 64K row packs a 30K PDF together with several short documents. What is true?**

- [ ] **A** Intra-document masking is what lets the PDF be seen in one piece <small>· look in [[2026-11-11 - Training Paradigms II#Long rows, at the end (Olmo 3)|TP II · Long rows, at the end]]</small>
- [ ] **B** Intra-document masking puts −∞ between tokens of different documents <small>· look in [[2026-11-11 - Training Paradigms II#Long rows, at the end (Olmo 3)|TP II · Long rows, at the end]]</small>
- [ ] **C** Without extra masking, tokens of a later short document could attend to the PDF <small>· look in [[2026-11-11 - Training Paradigms II#Long rows, at the end (Olmo 3)|TP II · Long rows, at the end]]</small>
- [ ] **D** With intra-document masking, the <|endoftext|> separators are no longer needed <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>

> [!success]- Answer (click to open)
> **B, C** · A ✗ · B ✓ · C ✓ · D ✗
>
> The causal mask alone lets every token see everything before it in the row. Intra-document masking blocks attention across documents. The PDF fits in one piece because the rows are 64K long; the separators still mark where documents end.

---

## Q15 · on-policy distillation
<small>TP II</small>

**On-policy distillation: for three tokens of the student’s answer, the student gives p = 0.60, 0.90, 0.30 and the teacher p = 0.70, 0.90, 0.02. What is true?**

- [ ] **A** The grades are held fixed, like rewards <small>· look in [[2026-11-11 - Training Paradigms II#B3 · On-policy distillation – a teacher, token by token|TP II §B3 On-policy distillation]]</small>
- [ ] **B** The teacher writes the answer and the student copies it <small>· look in [[2026-11-11 - Training Paradigms II#B3 · On-policy distillation – a teacher, token by token|TP II §B3 On-policy distillation]]</small>
- [ ] **C** The first token’s grade is about +0.15 <small>· look in [[2026-11-11 - Training Paradigms II#B3 · On-policy distillation – a teacher, token by token|TP II §B3 On-policy distillation]]</small>
- [ ] **D** The third token is pushed down <small>· look in [[2026-11-11 - Training Paradigms II#B3 · On-policy distillation – a teacher, token by token|TP II §B3 On-policy distillation]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> grade = log p_teacher − log p_student: log(0.7/0.6) ≈ +0.15, 0, log(0.02/0.3) ≈ −2.71. The student writes (on-policy); the teacher only scores, in one forward pass.

---

## Q16 · RLVR
<small>TP II</small>

**Which of these tasks can be trained with RLVR, i.e. with a program that checks the answer and returns 1 or 0?**

- [ ] **A** Summarise this article <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>
- [ ] **B** Comfort a sad friend <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>
- [ ] **C** Write a poem about cows <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>
- [ ] **D** Decide whether this joke is funny <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>

> [!success]- Answer (click to open)
> **none** · A ✗ · B ✗ · C ✗ · D ✗
>
> None: there is no program that can check any of these. They need RLHF or RLAIF (a reward model or a judge). RLVR fits maths answers, unit tests, checkable constraints, generated puzzles, valid JSON or tool calls.

---

## Q17 · agent loop and MCP
<small>Applied</small>

**What is true about the agent loop and MCP?**

- [ ] **A** An MCP host asks a server for its tools with tools/list and runs one with tools/call <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **B** The loop stops when the model’s reply contains no tool call <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **C** MCP servers’ tool descriptions bypass the prompt, so they cannot carry injected instructions <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **D** Each tool result is appended to the messages and read by the model as input <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> The loop: ask the model, run any tool calls, append the results, repeat until there are none (or max_steps). MCP is JSON-RPC over stdio or HTTP; the list and the results all go into the prompt, so they are untrusted input.

---

## Q18 · LLM as judge
<small>Applied</small>

**You use an LLM as a judge for your assistant’s answers. What is true?**

- [ ] **A** It tends to prefer longer answers <small>· look in [[2026-11-18 - Applied Generative AI I#3 · Who checks? Evals|Applied §3 Who checks? Evals]]</small>
- [ ] **B** It should score “I don’t know” below a wrong answer <small>· look in [[2026-11-18 - Applied Generative AI I#3 · Who checks? Evals|Applied §3 Who checks? Evals]]</small>
- [ ] **C** Swapping the order of the answers helps detect position bias <small>· look in [[2026-11-18 - Applied Generative AI I#3 · Who checks? Evals|Applied §3 Who checks? Evals]]</small>
- [ ] **D** It tends to prefer the answer shown first <small>· look in [[2026-11-18 - Applied Generative AI I#3 · Who checks? Evals|Applied §3 Who checks? Evals]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> Judge biases: position, length, its own style. Calibrate against human labels and swap the order. Score “I don’t know” above a wrong answer, or you select the best guesser.

---

## Q19 · context or weights?
<small>Applied</small>

**A course assistant must know this semester’s exam dates. What is true?**

- [ ] **A** Fine-tuning is good for format, style and behaviour <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **B** Putting the dates in the context, e.g. via RAG, is the right lever <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **C** With RAG, the answer can cite its source <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **D** Fine-tuning is the best way to teach it the new dates <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Fine-tuning is good for format and focus, bad for new knowledge: facts belong in the context. RAG is fresh after re-indexing and traceable to a source.

---

## Q20 · constrained decoding
<small>Applied</small>

**Constrained decoding enforces a JSON schema. What is true?**

- [ ] **A** At each step, tokens the schema does not allow get probability 0, and the rest are renormalised <small>· look in [[2026-11-18 - Applied Generative AI I#Station 6 · Write the answer (decode)|Applied · Station 6 · Write the answer]]</small>
- [ ] **B** Putting a “reasoning” field before the “date” field can make the date more often right <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **C** Every answer is valid JSON <small>· look in [[2026-11-18 - Applied Generative AI I#Station 6 · Write the answer (decode)|Applied · Station 6 · Write the answer]]</small>
- [ ] **D** Every date in the answer is correct <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Even the most likely token (```, 94 %) is struck out if the schema forbids it. Valid is not correct: 20/20 valid but 7/20 right dates. The model writes left to right, so reasoning first helped (20/20).

