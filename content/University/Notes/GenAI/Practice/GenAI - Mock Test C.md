---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Mock Test C

20 new questions, including some where no option is correct and one where all are.

> [!info] How the real test is scored (from the Preliminary Discussion)
> 20 questions · 40 minutes · closed book · **true multiple choice: 0 to 4 options may be correct**, and you are not told how many.
> Per question (1.5 points): **+1.5 ÷ n** for each correct tick (n = number of correct options), **−0.75** for each wrong tick, never below 0 per question. If no option is correct, **no tick at all = 1.5**.
> Tick the boxes (they are clickable in Obsidian), then open the collapsed answer. Score yourself with the key at the bottom.


## Q1 · the probabilistic view
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**What is true about the probabilistic view of AI?**

- [ ] **A** The goal of a generative model is p_θ(x) ≈ p(x)
- [ ] **B** Qwen3-0.6B-Base finds “The cow says moo” about four times as likely as “The cow says meow”
- [ ] **C** A discriminative model can generate new pictures by sampling p(cow | picture)
- [ ] **D** A discriminative model estimates p(picture)

> [!success]- Answer: A, B
> Regularities of the world show up as differences in probability. Discriminative: p(cow | picture), a score for a given input, nothing to sample pictures from. Generative: p(picture), the data itself.
> *Introduction to Generative AI · the probabilistic view*

## Q2 · BPE tokenization
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**A byte-level BPE vocabulary learned 256 ·c, 258 the, 260 ow, 262 ·the, 263 ·cow, 271 ·say, 272 ·café, 273 ·in. It tokenizes “the cows in the café say wow” as (258, 263, 115, 273, 262, 272, 271, 32, 119, 260). What is true?**

- [ ] **A** 32 is the byte for a space, kept as its own token here
- [ ] **B** <|endoftext|> was learned as a merge of frequent bytes
- [ ] **C** “·cows” is split into ·cow and s because no merge produced ·cows
- [ ] **D** “wow” is split because no merge covers ·wow as a whole

> [!success]- Answer: A, C, D
> Tokenizing applies the learned merges; whatever is not covered stays as smaller pieces, down to single bytes (115 = s, 119 = w, 32 = space). Special tokens are added after the merges, not learned from pairs.
> *Introduction to Generative AI · BPE tokenization*

## Q3 · symbolic vs subsymbolic
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**What is true about symbolic and subsymbolic AI?**

- [ ] **A** Symbolic AI learns its rules from data
- [ ] **B** A neural network is interpretable because each weight stands for a concept
- [ ] **C** Subsymbolic systems are corrected by editing their rules
- [ ] **D** A random forest is symbolic, because its trees are made of rules

> [!success]- Answer: none (no tick = full points)
> None. A random forest learns from data: subsymbolic. Subsymbolic models are corrected only by retraining; symbolic rules are written by people; neural network weights are hard to interpret. (No tick at all scores 1.5.)
> *Introduction to Generative AI · symbolic vs subsymbolic*

## Q4 · which parts mix tokens
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**In a decoder-only transformer, which parts let a token’s vector take in information from other positions?**

- [ ] **A** The feed-forward layer
- [ ] **B** LayerNorm
- [ ] **C** The token embedding
- [ ] **D** Masked multi-head attention

> [!success]- Answer: D
> Only attention compares and mixes tokens. Embedding (a lookup), feed-forward (each token alone) and LayerNorm (over one token’s own numbers) all work per token.
> *Theoretical Foundations of Language Modelling · which parts mix tokens*

## Q5 · dₖ and saturation
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**Two heads skip the √dₖ scaling. Head A has dₖ = 16, head B has dₖ = 1,024; q and k have entries of variance 1. What is true?**

- [ ] **A** Head B’s scores typically spread about 8 times wider than head A’s
- [ ] **B** Head B’s weights no longer add up to 1 per row
- [ ] **C** Less gradient flows back through head B’s softmax
- [ ] **D** Head B’s softmax is more likely to give one token nearly all the weight

> [!success]- Answer: A, C, D
> Standard deviation √dₖ: √1,024 / √16 = 32 / 4 = 8. Wide scores saturate the softmax: nearly one-hot, gradients ≈ 0. The softmax still sums to 1 in every case.
> *Theoretical Foundations of Language Modelling · dₖ and saturation*

## Q6 · the causal mask
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**A causal mask M is added to the scaled scores before the softmax. What is true?**

- [ ] **A** The diagonal is masked too, so a token cannot attend to itself
- [ ] **B** The mask is applied to V rather than to the scores
- [ ] **C** M holds −∞ strictly above the diagonal and 0 elsewhere
- [ ] **D** After the softmax, the first row is (1, 0, 0, …)

> [!success]- Answer: C, D
> Each token may see the tokens before it and itself. Row 1 has only itself, so all its weight goes there. The mask acts on the scores; e^(−∞) = 0 gives exactly zero weight to the future.
> *Theoretical Foundations of Language Modelling · the causal mask*

## Q7 · Add & Norm
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**A block computes x + Attention(Norm(x)), then x + FeedForward(Norm(x)). What is true?**

- [ ] **A** Attention adds what it found to the token’s vector, which is carried straight up
- [ ] **B** The norm is computed across all tokens of the sequence
- [ ] **C** The residual path helps gradients flow through deep stacks
- [ ] **D** Qwen3’s RMSNorm skips subtracting the mean

> [!success]- Answer: A, C, D
> This is the pre-norm residual stream. The norm works on each token’s own vector. RMSNorm only divides by the root mean square (no μ, no β).
> *Theoretical Foundations of Language Modelling · Add & Norm*

## Q8 · RoPE frequencies
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**RoPE turns pair i by θᵢ = b^(−2(i−1)/d) per position. What is true?**

- [ ] **A** b sets the maximum number of tokens the model can read
- [ ] **B** Qwen3 uses b = 1,000,000; the original value is 10,000
- [ ] **C** A larger base b makes the slowest pairs turn even more slowly
- [ ] **D** Pair 1 turns fastest

> [!success]- Answer: B, C, D
> θ₁ = b⁰ = 1 rad per position; later pairs are exponentially slower, more so for larger b. The base is not a hard length limit; context length depends on what was trained (and stretching like YaRN).
> *Theoretical Foundations of Language Modelling · RoPE frequencies*

## Q9 · H100 compute
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**An H100 delivers about 989 TFLOP/s in bf16 at peak. What is true?**

- [ ] **A** Model FLOPs utilisation is the share of the training tokens actually used
- [ ] **B** 40,000 H100-hours at 35 % give about 4.9 · 10²⁴ FLOPs
- [ ] **C** At 35 % model FLOPs utilisation, one H100-hour gives about 1.2 · 10¹⁸ useful FLOPs
- [ ] **D** One H100-hour at peak is about 3.56 · 10¹⁸ FLOPs

> [!success]- Answer: C, D
> 989 · 10¹² · 3,600 ≈ 3.56 · 10¹⁸; × 0.35 ≈ 1.23 · 10¹⁸; × 40,000 ≈ 4.9 · 10²², not 10²⁴. MFU is the share of the GPU’s peak spent on useful model maths; the rest goes to communication and memory waits.
> *Training Paradigms I · H100 compute*

## Q10 · cleaning and decontamination
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**What is true about preparing web data?**

- [ ] **A** Without it, benchmark scores can measure memorisation instead of ability
- [ ] **B** Deduplication removes exact copies only
- [ ] **C** A quality classifier can drop low-quality pages
- [ ] **D** Decontamination removes training pages that contain benchmark test items

> [!success]- Answer: A, C, D
> Filter (heuristics), deduplicate (exact copies, near copies, repeated passages), classify (topic, quality). The GSM8K problem posted on a forum with its answer gets dropped.
> *Training Paradigms I · cleaning and decontamination*

## Q11 · inputs and targets
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**A training step uses inputs = batch[:, :-1] and targets = batch[:, 1:]. What is true?**

- [ ] **A** A row of 4,097 token IDs gives 4,096 predictions
- [ ] **B** The model sees the target token as input at the same position
- [ ] **C** Positions labelled −100 are still counted in the loss, with weight 1
- [ ] **D** The target at position t is the token at position t + 1 of the row

> [!success]- Answer: A, D
> Inputs and targets are the same row shifted by one. With the causal mask, position t never sees token t + 1. −100 is PyTorch’s ignore label: those positions (e.g. image slots) get no loss.
> *Training Paradigms I · inputs and targets*

## Q12 · parallelism
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**Training runs on many GPUs. What travels between them?**

- [ ] **A** Data parallel: gradients, averaged once per step
- [ ] **B** Tensor parallel: partial results, summed after every layer
- [ ] **C** Pipeline parallel: nothing, because each GPU holds all layers
- [ ] **D** Expert parallel: tokens, sent to their experts and back

> [!success]- Answer: A, B, D
> Pipeline parallel splits the model by layers: activations travel forward and their gradients back. FSDP / ZeRO-3 gathers each layer’s weights when needed and frees them afterwards.
> *Training Paradigms I · parallelism*

## Q13 · MLA and GQA
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**What is true about grouped-query attention (GQA) and multi-head latent attention (MLA)?**

- [ ] **A** Both make the KV cache smaller
- [ ] **B** In GQA several query heads share one key/value head
- [ ] **C** MLA caches a small latent per token (512 numbers in DeepSeek-V3) instead of full keys and values
- [ ] **D** In MLA, W^UK can be folded into the query, so the full keys need not be built

> [!success]- Answer: A, B, C, D
> MLA: C = X W^DKV (7,168 → 512); K = C W^UK and V = C W^UV (512 → 16,384). qₜkⱼᵀ = (qₜ W^UKᵀ) cⱼᵀ. 512 vs 2 · 16,384: 64× less cache.
> *Training Paradigms II · MLA and GQA*

## Q14 · packing long rows
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**In the long-context stage, a 64K row packs a 30K PDF together with several short documents. What is true?**

- [ ] **A** Intra-document masking is what lets the PDF be seen in one piece
- [ ] **B** Intra-document masking puts −∞ between tokens of different documents
- [ ] **C** Without extra masking, tokens of a later short document could attend to the PDF
- [ ] **D** With intra-document masking, the <|endoftext|> separators are no longer needed

> [!success]- Answer: B, C
> The causal mask alone lets every token see everything before it in the row. Intra-document masking blocks attention across documents. The PDF fits in one piece because the rows are 64K long; the separators still mark where documents end.
> *Training Paradigms II · packing long rows*

## Q15 · on-policy distillation
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**On-policy distillation: for three tokens of the student’s answer, the student gives p = 0.60, 0.90, 0.30 and the teacher p = 0.70, 0.90, 0.02. What is true?**

- [ ] **A** The grades are held fixed, like rewards
- [ ] **B** The teacher writes the answer and the student copies it
- [ ] **C** The first token’s grade is about +0.15
- [ ] **D** The third token is pushed down

> [!success]- Answer: A, C, D
> grade = log p_teacher − log p_student: log(0.7/0.6) ≈ +0.15, 0, log(0.02/0.3) ≈ −2.71. The student writes (on-policy); the teacher only scores, in one forward pass.
> *Training Paradigms II · on-policy distillation*

## Q16 · RLVR
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**Which of these tasks can be trained with RLVR, i.e. with a program that checks the answer and returns 1 or 0?**

- [ ] **A** Summarise this article
- [ ] **B** Comfort a sad friend
- [ ] **C** Write a poem about cows
- [ ] **D** Decide whether this joke is funny

> [!success]- Answer: none (no tick = full points)
> None: there is no program that can check any of these. They need RLHF or RLAIF (a reward model or a judge). RLVR fits maths answers, unit tests, checkable constraints, generated puzzles, valid JSON or tool calls.
> *Training Paradigms II · RLVR*

## Q17 · agent loop and MCP
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**What is true about the agent loop and MCP?**

- [ ] **A** An MCP host asks a server for its tools with tools/list and runs one with tools/call
- [ ] **B** The loop stops when the model’s reply contains no tool call
- [ ] **C** MCP servers’ tool descriptions bypass the prompt, so they cannot carry injected instructions
- [ ] **D** Each tool result is appended to the messages and read by the model as input

> [!success]- Answer: A, B, D
> The loop: ask the model, run any tool calls, append the results, repeat until there are none (or max_steps). MCP is JSON-RPC over stdio or HTTP; the list and the results all go into the prompt, so they are untrusted input.
> *Applied Generative AI I · agent loop and MCP*

## Q18 · LLM as judge
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**You use an LLM as a judge for your assistant’s answers. What is true?**

- [ ] **A** It tends to prefer longer answers
- [ ] **B** It should score “I don’t know” below a wrong answer
- [ ] **C** Swapping the order of the answers helps detect position bias
- [ ] **D** It tends to prefer the answer shown first

> [!success]- Answer: A, C, D
> Judge biases: position, length, its own style. Calibrate against human labels and swap the order. Score “I don’t know” above a wrong answer, or you select the best guesser.
> *Applied Generative AI I · LLM as judge*

## Q19 · context or weights?
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**A course assistant must know this semester’s exam dates. What is true?**

- [ ] **A** Fine-tuning is good for format, style and behaviour
- [ ] **B** Putting the dates in the context, e.g. via RAG, is the right lever
- [ ] **C** With RAG, the answer can cite its source
- [ ] **D** Fine-tuning is the best way to teach it the new dates

> [!success]- Answer: A, B, C
> Fine-tuning is good for format and focus, bad for new knowledge: facts belong in the context. RAG is fresh after re-indexing and traceable to a source.
> *Applied Generative AI I · context or weights?*

## Q20 · constrained decoding
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**Constrained decoding enforces a JSON schema. What is true?**

- [ ] **A** At each step, tokens the schema does not allow get probability 0, and the rest are renormalised
- [ ] **B** Putting a “reasoning” field before the “date” field can make the date more often right
- [ ] **C** Every answer is valid JSON
- [ ] **D** Every date in the answer is correct

> [!success]- Answer: A, B, C
> Even the most likely token (```, 94 %) is struck out if the schema forbids it. Valid is not correct: 20/20 valid but 7/20 right dates. The model writes left to right, so reasoning first helped (20/20).
> *Applied Generative AI I · constrained decoding*

---

> [!example]- Answer key
>
> | Q | Correct | Lecture |
> |---|---|---|
> | 1 | A, B | Introduction to Generative AI |
> | 2 | A, C, D | Introduction to Generative AI |
> | 3 | none | Introduction to Generative AI |
> | 4 | D | Theoretical Foundations of Language Modelling |
> | 5 | A, C, D | Theoretical Foundations of Language Modelling |
> | 6 | C, D | Theoretical Foundations of Language Modelling |
> | 7 | A, C, D | Theoretical Foundations of Language Modelling |
> | 8 | B, C, D | Theoretical Foundations of Language Modelling |
> | 9 | C, D | Training Paradigms I |
> | 10 | A, C, D | Training Paradigms I |
> | 11 | A, D | Training Paradigms I |
> | 12 | A, B, D | Training Paradigms I |
> | 13 | A, B, C, D | Training Paradigms II |
> | 14 | B, C | Training Paradigms II |
> | 15 | A, C, D | Training Paradigms II |
> | 16 | none | Training Paradigms II |
> | 17 | A, B, D | Applied Generative AI I |
> | 18 | A, C, D | Applied Generative AI I |
> | 19 | A, B, C | Applied Generative AI I |
> | 20 | A, B, C | Applied Generative AI I |
