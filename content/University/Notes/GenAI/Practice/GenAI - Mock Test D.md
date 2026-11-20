---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Mock Test D

20 questions in the format of the real test, 40 minutes. The questions of Mock Tests A–D are **not** in the lecture notes, so each test stays fresh.

> [!info] How the test is scored
> 20 questions · 40 minutes · closed book · **0 to 4 options may be correct**, and you are not told how many. Treat every option as its own true/false question.
> Per question (1.5 points): **+1.5 ÷ n** per correct tick (n = number of correct options), **−0.75** per wrong tick, never below 0. If no option is correct, **no tick = 1.5**.
> Next to every option: the section of the notes where the answer is.

---

## Q1 · autoregression
<small>Intro</small>

**A language model generates text autoregressively. What is true?**

- [ ] **A** Each new token is drawn from p(xₜ | x₁ … xₜ₋₁) and appended before the next step <small>· look in [[2026-10-21 - Introduction to Generative AI#4 · Autoregressive – piece by piece|Intro §4 Autoregressive]]</small>
- [ ] **B** Writing p(x₁, …, x_T) as a product of next-token probabilities is only an approximation <small>· look in [[2026-10-21 - Introduction to Generative AI#4 · Autoregressive – piece by piece|Intro §4 Autoregressive]]</small>
- [ ] **C** “Auto” means each value is predicted from the model’s own previous outputs <small>· look in [[2026-10-21 - Introduction to Generative AI#4 · Autoregressive – piece by piece|Intro §4 Autoregressive]]</small>
- [ ] **D** It needs a separately trained model for every position t <small>· look in [[2026-10-21 - Introduction to Generative AI#4 · Autoregressive – piece by piece|Intro §4 Autoregressive]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> The chain rule is exact for any joint distribution, so one next-token distribution is all the model has to learn. Generation: draw, append, ask again.

---

## Q2 · Bayes on the farm
<small>Intro</small>

**On the farm, p(moo | cow) = 35 %, p(cow) = 40 % and p(moo) = 16 %. What is true?**

- [ ] **A** p(cow | moo) = 0.35 · 0.40 / 0.16 ≈ 0.875 <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>
- [ ] **B** p(cow, moo) = 35 % <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>
- [ ] **C** The cow row of the joint table (moo, woof, meow, nothing) adds up to p(cow) = 40 % <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>
- [ ] **D** p(nothing) = 24 % <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> Bayes: p(cow | moo) = p(moo | cow) p(cow) / p(moo) = 0.875. The joint is p(cow) · p(moo | cow) = 0.14. A marginal sums a row or column: cow = 14 + 1 + 1 + 24 = 40 %; nothing = 24 + 16 + 18 = 58 %.

---

## Q3 · noise in diffusion
<small>Intro</small>

**What plays the role of noise when diffusion is used for different kinds of data?**

- [ ] **A** Gaussian noise for pixels and sound <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>
- [ ] **B** Masks: tokens replaced by [MASK] and un-masked step by step, for text <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>
- [ ] **C** Gaussian noise added to the token IDs, for text <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>
- [ ] **D** Categorical noise on the probabilities of the bases, for DNA <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Pixels, sound and video get Gaussian noise; text uses masks (masked diffusion); DNA uses probabilities. Token IDs are addresses, adding noise to them means nothing.

---

## Q4 · embeddings
<small>Intro</small>

**What is true about word embeddings?**

- [ ] **A** Cosine similarity of ·cow with ·cows is about 0.75, with ·piano about 0 <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **B** king − man + woman lands near queen <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **C** Word2Vec learns its vectors by predicting the next sentence <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **D** They are useful only for generation, not for search <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Near means similar, directions carry relations. Word2Vec predicts a missing word from its neighbours (“you shall know a word by the company it keeps”). Uses: semantic search, RAG, recommendations, clustering.

---

## Q5 · LayerNorm by hand
<small>LM</small>

**LayerNorm is applied to one token’s vector x = (1, 3), with γ = 1, β = 0 and ε negligible. What is true?**

- [ ] **A** μ = 2 and σ² = 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **B** The output is (−1, 1) <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **C** The output depends on the other tokens of the sequence <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **D** Multiplying x by 10 changes the output <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> μ = (1 + 3)/2 = 2, σ² = (1 + 1)/2 = 1, so (x − μ)/σ = (−1, 1). For (10, 30): μ = 20, σ = 10, again (−1, 1). Mean and variance are taken over this one token’s numbers only.

---

## Q6 · relative position bias
<small>LM</small>

**Relative position bias (T5-style) adds a number b to each attention score. What is true?**

- [ ] **A** There is one learned number per distance between query and key <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>
- [ ] **B** It rotates queries and keys by an angle <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>
- [ ] **C** Two pairs of tokens with the same distance get the same bias <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>
- [ ] **D** It is added to the token embeddings before the first block <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> Relative bias: scores + B, with B depending only on the distance. Rotating q and k is RoPE; adding to the embeddings is what absolute encodings do.

---

## Q7 · one masked row
<small>LM</small>

**Masked attention for token 3 of “The small purple cow” (one head). What is true?**

- [ ] **A** The weight for “cow” (position 4) is exactly 0 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>
- [ ] **B** The weights for positions 1 to 3 add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **C** Token 3’s output is a weighted sum of the queries <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **D** The score for key 2 is q₃ · k₂ / √dₖ <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> The mask puts −∞ on later positions, e^−∞ = 0, and softmax shares the whole 1 among positions 1–3. The output mixes the values, weighted by the scores of query 3 against each key.

---

## Q8 · size of the feed-forward
<small>LM</small>

**The original transformer has d_model = 512 and a feed-forward of width 2,048. What is true?**

- [ ] **A** The feed-forward holds about 2 · 512 · 2,048 ≈ 2.1M weights <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **B** The attention projections W^Q, W^K, W^V, W^O hold about 4 · 512² ≈ 1.05M weights <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **C** The feed-forward is applied once to the whole sequence as one long vector <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **D** ReLU sets the negative values to 0 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Up 512 → 2,048 and down 2,048 → 512: twice 512 · 2,048. That is most of a block’s weights. The same small network runs on each token on its own.

---

## Q9 · Chinchilla by hand
<small>TP I</small>

**A lab has C = 1.2 · 10²³ FLOPs and trains compute-optimally (D ≈ 20 N). What is true?**

- [ ] **A** N ≈ √(C / 120) ≈ 32B parameters <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **B** D ≈ 0.63T tokens <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **C** Doubling C doubles the optimal N <small>· look in [[2026-11-04 - Training Paradigms I#1.2 Compute-optimal (Chinchilla, Hoffmann et al. 2022)|TP I §1.2 Compute-optimal]]</small>
- [ ] **D** 6 · 32B · 0.63T gives back about 1.2 · 10²³ <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> C ≈ 6ND = 6N · 20N = 120N², so N = √(10²¹) ≈ 3.2 · 10¹⁰ and D = 20N ≈ 6.3 · 10¹¹. N and D both grow like √C, so doubling C multiplies N by about 1.4.

---

## Q10 · vocabulary cost
<small>TP I</small>

**The pilot (d = 2,048, 2B active) would use a vocabulary of V = 100k, embedding and output layer untied. What is true?**

- [ ] **A** Embedding plus output layer hold about 0.41B weights <small>· look in [[2026-11-04 - Training Paradigms I#1.8 What a vocabulary costs|TP I §1.8 What a vocabulary costs]]</small>
- [ ] **B** That is about 20 % of the active parameters <small>· look in [[2026-11-04 - Training Paradigms I#1.8 What a vocabulary costs|TP I §1.8 What a vocabulary costs]]</small>
- [ ] **C** Tying embedding and output layer would double that cost <small>· look in [[2026-11-04 - Training Paradigms I#1.8 What a vocabulary costs|TP I §1.8 What a vocabulary costs]]</small>
- [ ] **D** A larger vocabulary always lowers the cost per sentence <small>· look in [[2026-11-04 - Training Paradigms I#1.7 Tokenizer – what a language costs|TP I §1.7 Tokenizer]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> 2 · V · d = 2 · 100,000 · 2,048 ≈ 0.41B, about 20 % of 2B. Tying shares one table, so it halves the cost. A bigger vocabulary means fewer tokens for well-covered languages but bigger matrices: not “always” cheaper.

---

## Q11 · the data pipeline
<small>TP I</small>

**What is true about preparing pre-training data?**

- [ ] **A** The mix is chosen by training many small proxy models on different mixes and predicting the score <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **B** Common Crawl text is used as it comes, without filtering <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **C** Near copies and repeated passages are removed too, not only exact copies <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **D** Decontamination removes benchmark items from the benchmark itself <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> Filters → deduplicate (exact, near, repeated passages) → classify. Decontamination removes training pages that contain test items. The mix comes from proxy models and a fitted predictor.

---

## Q12 · one pre-training step
<small>TP I</small>

**The pilot trains on batches of 1,024 rows × 4,096 tokens. What is true?**

- [ ] **A** One step sees about 4.2M tokens <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **B** The loss of the step is the mean of about 4.2M token losses <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **C** A token given p = 0.000024 has a loss of about 10.7 <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **D** The optimiser updates the weights after every single token <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> 1,024 · 4,096 ≈ 4.2M. Loss per token = −log p: −log 0.000024 ≈ 10.7. One number for the whole batch, then one backward pass and one update per step.

---

## Q13 · KV cache by hand
<small>TP II</small>

**A model has 32 layers, 8 key/value heads of size 128 and stores the cache in bf16 (2 bytes). What is true?**

- [ ] **A** The cache grows by 2 · 32 · 8 · 128 · 2 bytes = 128 KiB per token <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **B** With 32 key/value heads instead of 8, the cache would be 4× larger <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **C** Prefill fills the cache one prompt token per step <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **D** The bytes per token grow with the length of the context <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> 2 (K and V) × layers × KV heads × head size × bytes = 131,072 B. The per-token size is fixed; the total grows with the context. Prefill reads the whole prompt in parallel.

---

## Q14 · DPO by hand
<small>TP II</small>

**DPO with β = 0.1. Chosen: policy −20, reference −25. Rejected: policy −30, reference −28. What is true?**

- [ ] **A** The bracket is 0.1 · [5 − (−2)] = 0.7 <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>
- [ ] **B** The loss is −log σ(0.7) ≈ 0.40 <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>
- [ ] **C** DPO samples fresh answers from the policy during training <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>
- [ ] **D** A larger β lets the policy drift further from the reference <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> gain⁺ = −20 − (−25) = +5, gain⁻ = −30 − (−28) = −2. σ(0.7) ≈ 0.67, −log 0.67 ≈ 0.40. DPO is offline (fixed pairs); a larger β keeps the policy closer to the reference.

---

## Q15 · RLVR and RLAIF
<small>TP II</small>

**What is true about where the reward comes from?**

- [ ] **A** In RLAIF, an AI judge labels the answers, guided by a constitution written by people <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **B** In RLVR, a program returns 1 or 0 <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>
- [ ] **C** RLVR needs a separately trained reward model <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>
- [ ] **D** “Answer in under 50 words, without commas” can be checked by a script <small>· look in [[2026-11-11 - Training Paradigms II#B5 · RLVR – a checker|TP II §B5 RLVR]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> RLVR: verifiable rewards from a checker (maths, unit tests, instruction constraints, format). RLAIF replaces human labels with an AI judge, at under $0.01 per label.

---

## Q16 · GRPO edge cases
<small>TP II</small>

**GRPO samples 4 answers to one prompt, with rewards 1, 1, 0, 0. What is true?**

- [ ] **A** The advantages are +1, +1, −1, −1 <small>· look in [[2026-11-11 - Training Paradigms II#B8 · GRPO (DeepSeekMath)|TP II §B8 GRPO]]</small>
- [ ] **B** If all four answers got reward 1, nothing would be learned from this prompt <small>· look in [[2026-11-11 - Training Paradigms II#B8 · GRPO (DeepSeekMath)|TP II §B8 GRPO]]</small>
- [ ] **C** GRPO needs a value model to compute these advantages <small>· look in [[2026-11-11 - Training Paradigms II#PPO vs GRPO|TP II · PPO vs GRPO]]</small>
- [ ] **D** PPO’s clip with ε = 0.2 lets the ratio ρ go up to 1.5 <small>· look in [[2026-11-11 - Training Paradigms II#B7 · PPO|TP II §B7 PPO]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> mean = 0.5, std = 0.5, so A = (r − 0.5)/0.5 = ±1. Equal rewards give std = 0 and A = 0 everywhere. The group mean replaces the value model. Clip keeps ρ in [0.8, 1.2].

---

## Q17 · what a request costs
<small>Applied</small>

**A provider charges 1 USD per million input tokens and 5 USD per million output tokens. A request has 1,000 input and 200 output tokens. What is true?**

- [ ] **A** Without caching it costs 0.2 US cents <small>· look in [[2026-11-18 - Applied Generative AI I#Station 7 · Stream back|Applied · Station 7 · Stream back]]</small>
- [ ] **B** Output tokens are cheaper than input tokens <small>· look in [[2026-11-18 - Applied Generative AI I#Station 7 · Stream back|Applied · Station 7 · Stream back]]</small>
- [ ] **C** Caching part of the input lowers only the input part of the bill <small>· look in [[2026-11-18 - Applied Generative AI I#Station 7 · Stream back|Applied · Station 7 · Stream back]]</small>
- [ ] **D** Thinking tokens are billed as output tokens <small>· look in [[2026-11-18 - Applied Generative AI I#Station 7 · Stream back|Applied · Station 7 · Stream back]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> 1,000 · $1/M = 0.1 ¢ and 200 · $5/M = 0.1 ¢. Output costs about 5× input and includes the thinking. A cached prefix costs 2.5–10 % of the input price; the output price stays.

---

## Q18 · chat memory and the prefix cache
<small>Applied</small>

**A chat app talks to an API that offers prefix caching. What is true?**

- [ ] **A** Turn 5 of a chat sends the earlier turns again <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **B** Putting what stays the same (system prompt, documents) first makes better use of the cache <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>
- [ ] **C** One changed token at the end of the prompt makes the server read the whole prompt again <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>
- [ ] **D** Prefill time grows mainly with the length of the answer <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> The API is stateless: memory is the resent history. The prefix cache matches up to the first difference, so a change at the end costs little and one at the start costs everything after it. Prompt length drives the wait before the first token.

---

## Q19 · agents in practice
<small>Applied</small>

**You let a coding agent run for an afternoon. What is true?**

- [ ] **A** 93 % of permission prompts get approved: people stop reading them <small>· look in [[2026-11-18 - Applied Generative AI I#4 · Let it run (agents in practice)|Applied §4 Let it run]]</small>
- [ ] **B** Limiting file access without limiting network access is not enough <small>· look in [[2026-11-18 - Applied Generative AI I#4 · Let it run (agents in practice)|Applied §4 Let it run]]</small>
- [ ] **C** Compaction keeps every detail of the earlier history <small>· look in [[2026-11-18 - Applied Generative AI I#4 · Let it run (agents in practice)|Applied §4 Let it run]]</small>
- [ ] **D** Quality can drop long before the context window is full <small>· look in [[2026-11-18 - Applied Generative AI I#4 · Let it run (agents in practice)|Applied §4 Let it run]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Give the agent a box: limit files and network together. The window fills with tool calls and results; compaction summarises and loses details.

---

## Q20 · how RAG fails
<small>Applied</small>

**A course assistant uses RAG. What is true?**

- [ ] **A** Facts in the middle of a long context are missed more often than at the start or end <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **B** When retrieval misses, the model may answer from its weights just as confidently <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **C** Adding more passages always improves the answers <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **D** recall@k measures how often the final answer is correct <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Lost in the middle (Liu et al. 2023); context rot: more is not better. recall@k measures retrieval: is the right passage among the top k?

