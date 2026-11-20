---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Mock Test B

20 questions in the format of the real test, 40 minutes. The questions of Mock Tests A–D are **not** in the lecture notes, so each test stays fresh.

> [!info] How the test is scored
> 20 questions · 40 minutes · closed book · **0 to 4 options may be correct**, and you are not told how many. Treat every option as its own true/false question.
> Per question (1.5 points): **+1.5 ÷ n** per correct tick (n = number of correct options), **−0.75** per wrong tick, never below 0. If no option is correct, **no tick = 1.5**.
> Next to every option: the section of the notes where the answer is.

---

## Q1 · symbolic AI
<small>Intro</small>

**A system uses the rule “if horns and udder and hooves and tail then cow”. What is true?**

- [ ] **A** It classifies a dehorned cow as not a cow <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **B** It is an example of subsymbolic AI <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **C** A neural network trained on cow pictures can be fixed the same way, by editing one weight <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **D** Its mistake can be fixed directly by editing the rule <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> Symbolic systems reason only within explicitly encoded patterns, so unseen cases break them; but the rules are readable and editable. Subsymbolic models can only be corrected indirectly, by retraining on other data.

---

## Q2 · sound as tokens
<small>Intro</small>

**Sound is turned into tokens. What is true?**

- [ ] **A** 2 seconds at 24 kHz are 48,000 numbers <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>
- [ ] **B** Residual vector quantisation applies a further codebook to what the previous one left out <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>
- [ ] **C** T seconds at sample rate f_s give a vector x with f_s · T numbers <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>
- [ ] **D** Each audio sample becomes one token <small>· look in [[2026-10-21 - Introduction to Generative AI#6 · Tokens for every modality|Intro §6 Tokens for every modality]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> A window of samples goes through a learned encoder to one vector zₜ; optionally it is snapped to the nearest codebook entry, often repeated on the remainder (EnCodec). One token per window, not per sample.

---

## Q3 · contextual embeddings
<small>Intro</small>

**Qwen3-0.6B reads “The cat says meow. The cow says”. What is true about the vector of the last token, ·says?**

- [ ] **A** Only the last layers turn it into a confident prediction of ·moo <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **B** Already a few layers in, it reads as ·moo with high probability <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **C** Right after the embedding lookup, it is the same vector as ·says in any other sentence <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **D** The embedding table is where the context changes it <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> The lookup is static. Read through the final layer, layer 25 still says ·cow; ·moo appears at layer 26 and reaches 93.5 % at layer 28. The context comes in through attention, not through the table.

---

## Q4 · mask without positional encoding
<small>LM</small>

**A decoder has causal masked attention but no positional encoding. It reads “dog bites man” and “man bites dog” (one token per word). What is true?**

- [ ] **A** The output for the first token depends only on that token <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **B** The outputs are the same vectors as before, only in permuted order <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **C** “dog” gets the same output vector in both sentences <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **D** In principle the model can tell positions apart, since each position sees a different number of tokens <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> The permutation argument holds only without the mask. With it, “dog” first sees only itself; “dog” last sees all three words. That is why the example test said: without positional encoding and without a mask.

---

## Q5 · softmax
<small>LM</small>

**One row of scaled scores is s = (2, 1, 0). What is true about the softmax weights?**

- [ ] **A** The largest score gets the largest weight <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **B** Adding 5 to every score changes the weights <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **C** They add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **D** Multiplying every score by 10 brings the weights closer to (1, 0, 0) <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> softmax(s)ⱼ = e^sⱼ / Σ e^sᵢ: positive, sums to 1, keeps the order. A constant shift cancels out. Larger scores sharpen it toward one-hot: exactly why scores are divided by √dₖ.

---

## Q6 · shapes in attention
<small>LM</small>

**X has shape n × d_model; W^Q and W^K are d_model × dₖ; W^V is d_model × d_v. What is true?**

- [ ] **A** QKᵀ has shape dₖ × dₖ <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **B** QKᵀ has shape n × n <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **C** The output softmax(QKᵀ/√dₖ)V has shape n × d_v <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>
- [ ] **D** Row i of the weights says how much token i takes from each token j <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Step by step with shapes (one head, n tokens)|LM · Step by step with shapes]]</small>

> [!success]- Answer (click to open)
> **B, C, D** · A ✗ · B ✓ · C ✓ · D ✓
>
> Q and K are n × dₖ, so QKᵀ is n × n: a score for every pair. Softmax runs along each row (over j). Weights (n × n) times V (n × d_v) give one new vector per token.

---

## Q7 · positional encodings
<small>LM</small>

**Learned absolute vs sinusoidal positional encodings. What is true?**

- [ ] **A** With sinusoidal encodings, k positions later is the same fixed turn of the vector from any position <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>
- [ ] **B** A learned table with 1,024 rows has no vector for position 1,024 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>
- [ ] **C** Sinusoidal encodings have no parameters to learn <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Absolute vs relative|LM · Absolute vs relative]]</small>
- [ ] **D** Both are applied by rotating queries and keys <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> GPT-2’s learned table: fixed context length, late rows seen only in long texts. Sinusoidal: computed for any position; distances only implicit. Both are added to the token vectors; rotating q and k is RoPE.

---

## Q8 · output layer
<small>LM</small>

**After the last block, “The cow says” gives a matrix H of shape 3 × 1,024 in Qwen3-0.6B. What is true?**

- [ ] **A** In Qwen3-0.6B the output layer shares its weights with the embedding table <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#6 · Linear + softmax – what comes next|LM §6 Linear + softmax]]</small>
- [ ] **B** To generate the next token, only the last row is needed <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#6 · Linear + softmax – what comes next|LM §6 Linear + softmax]]</small>
- [ ] **C** A linear layer maps each row to 151,936 logits <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#6 · Linear + softmax – what comes next|LM §6 Linear + softmax]]</small>
- [ ] **D** Softmax turns each row into a distribution over the next token <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#6 · Linear + softmax – what comes next|LM §6 Linear + softmax]]</small>

> [!success]- Answer (click to open)
> **A, B, C, D** · A ✓ · B ✓ · C ✓ · D ✓
>
> Linear d → V gives one number per vocabulary entry; softmax makes them probabilities. Training scores every row (each predicts the token after it); generation uses only the last. Qwen3-0.6B ties the head to the embedding table.

---

## Q9 · scaling laws
<small>TP I</small>

**What is true about scaling laws?**

- [ ] **A** Loss falls as a power law in compute: a straight line on log–log axes <small>· look in [[2026-11-04 - Training Paradigms I#1.1 Scaling laws (Kaplan et al. 2020)|TP I §1.1 Scaling laws]]</small>
- [ ] **B** When compute grows, compute-optimal parameters and tokens grow by roughly the same factor <small>· look in [[2026-11-04 - Training Paradigms I#1.2 Compute-optimal (Chinchilla, Hoffmann et al. 2022)|TP I §1.2 Compute-optimal]]</small>
- [ ] **C** Chinchilla: compute-optimal training uses about 20 tokens per parameter <small>· look in [[2026-11-04 - Training Paradigms I#1.2 Compute-optimal (Chinchilla, Hoffmann et al. 2022)|TP I §1.2 Compute-optimal]]</small>
- [ ] **D** The Chinchilla-optimal model also minimises the cost of serving it <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Kaplan: log-linear trends in C, D, N. Chinchilla: N_opt ∝ C^0.49, D_opt ∝ C^0.51. It optimises training compute only: serving a bigger model costs more, which is why the pilot over-trains a small one.

---

## Q10 · mixture of experts
<small>TP I</small>

**The pilot activates 1 shared + 8 routed experts out of 256 routed, each 768 wide, with d = 2,048. What is true?**

- [ ] **A** It has about 30B parameters in total but per-token compute like a 2B model <small>· look in [[2026-11-04 - Training Paradigms I#1.6 Dense vs mixture of experts|TP I §1.6 Dense vs mixture of experts]]</small>
- [ ] **B** The active feed-forward width is 9 · 768 = 6,912, about 3.4 · d <small>· look in [[2026-11-04 - Training Paradigms I#1.6 Dense vs mixture of experts|TP I §1.6 Dense vs mixture of experts]]</small>
- [ ] **C** The router picks the experts once per document <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>
- [ ] **D** During training every token goes through all 256 experts <small>· look in [[2026-11-04 - Training Paradigms I#1.6 Dense vs mixture of experts|TP I §1.6 Dense vs mixture of experts]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Only the selected experts run; the router decides per token (and per layer). Total ÷ active ≈ 32 keeps the knowledge capacity of a much larger model at the compute of the active part.

---

## Q11 · the loss
<small>TP I</small>

**The training loss per token is −log p, with p the probability given to the true next token. What is true?**

- [ ] **A** A model that spreads probability evenly over a 151,936-token vocabulary has a loss of about 12 <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **B** p = 0.5 gives a loss of about 0.69 <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **C** The batch loss is the sum of the GPUs’ mean losses <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **D** p = 1 gives a loss of 0 <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> −log 1 = 0, −log 0.5 = ln 2 ≈ 0.69, −log(1/151,936) = ln 151,936 ≈ 11.9. The batch loss is the mean over all tokens; with equal counts per GPU that is the mean of the GPUs’ means, not their sum.

---

## Q12 · gradient clipping
<small>TP I</small>

**Gradients are clipped to norm 1.0. What is true?**

- [ ] **A** A gradient of norm 0.5 is left as it is <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **B** A gradient of norm 8 is scaled down to norm 1, keeping its direction <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **C** Each weight’s gradient is clipped to [−1, 1] on its own <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>
- [ ] **D** It guards against the huge gradient of a loss spike <small>· look in [[2026-11-04 - Training Paradigms I#3 · Pre-training – one step|TP I §3 Pre-training]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> The norm of the gradient of all weights together is computed; if it exceeds 1.0, the whole vector is rescaled. Usual steps are untouched; a spike step keeps its direction but gets length 1.

---

## Q13 · mid-training
<small>TP II</small>

**What is part of mid-training?**

- [ ] **A** Finishing on the best data, including maths and reasoning <small>· look in [[2026-11-11 - Training Paradigms II#A2 · Finish on the best data|TP II §A2 Finish on the best data]]</small>
- [ ] **B** Extending the context to longer rows <small>· look in [[2026-11-11 - Training Paradigms II#A3 · Extend the context – 4,096 → 1,000,000|TP II §A3 Extend the context]]</small>
- [ ] **C** Adding thinking traces as chat conversations with user and assistant roles <small>· look in [[2026-11-11 - Training Paradigms II#A2 · Finish on the best data|TP II §A2 Finish on the best data]]</small>
- [ ] **D** Cooling down the learning rate <small>· look in [[2026-11-11 - Training Paradigms II#A1 · Cool down the learning rate|TP II §A1 Cool down the learning rate]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> Mid-training: cool down, finish on the best data, extend the context. Thinking traces go in as plain text without chat format; the chat template comes with post-training.

---

## Q14 · YaRN
<small>TP II</small>

**Olmo 3 7B (RoPE base 500,000) is extended from 8K to 64K with YaRN, scale 8. What is true?**

- [ ] **A** Dividing every pair by 8 instead would blur the order of nearby tokens <small>· look in [[2026-11-11 - Training Paradigms II#Stretching the positions – YaRN|TP II · Stretching the positions]]</small>
- [ ] **B** The fastest RoPE pairs are left unchanged <small>· look in [[2026-11-11 - Training Paradigms II#Stretching the positions – YaRN|TP II · Stretching the positions]]</small>
- [ ] **C** The slowest pairs are slowed down by a factor of 8 <small>· look in [[2026-11-11 - Training Paradigms II#Stretching the positions – YaRN|TP II · Stretching the positions]]</small>
- [ ] **D** With YaRN, no training on long documents is needed <small>· look in [[2026-11-11 - Training Paradigms II#Long rows, at the end (Olmo 3)|TP II · Long rows, at the end]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Pair 1 turns 1,304 times in 8K: every angle seen, keep it. Pair 48 turns only 31°: beyond 8K it would reach unseen angles, so ÷8 maps 0–64K into the seen range. A long-context stage (64K rows) still follows.

---

## Q15 · the baseline
<small>TP II</small>

**Four answers get rewards 0.8, 0.5, 0.3 and 0.1. Compare L = −r · Σ log p with L = −(r − b) · Σ log p, b = 0.425. What is true?**

- [ ] **A** Without the baseline, all four answers are made more likely <small>· look in [[2026-11-11 - Training Paradigms II#B6 · Why not just the reward? The baseline|TP II §B6 Why not just the reward? The baseline]]</small>
- [ ] **B** In PPO, the baseline is the value model’s prediction Vₜ <small>· look in [[2026-11-11 - Training Paradigms II#B7 · PPO|TP II §B7 PPO]]</small>
- [ ] **C** The baseline changes which answer has the highest reward <small>· look in [[2026-11-11 - Training Paradigms II#B6 · Why not just the reward? The baseline|TP II §B6 Why not just the reward? The baseline]]</small>
- [ ] **D** With the baseline, answers below the average are made less likely <small>· look in [[2026-11-11 - Training Paradigms II#B6 · Why not just the reward? The baseline|TP II §B6 Why not just the reward? The baseline]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> All rewards are positive, so −r · Σ log p pushes every answer up, the bad ones just less. r − b is negative for 0.3 and 0.1. Subtracting a constant does not reorder rewards. PPO: Aₜ = Gₜ − Vₜ; GRPO: the group mean.

---

## Q16 · reward model
<small>TP II</small>

**A reward model is trained on people’s rankings of 4 answers to one prompt. What is true?**

- [ ] **A** Each pair’s loss is −log σ(r⁺ − r⁻) <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **B** During RL, the reward model keeps training alongside the policy <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **C** The ranking gives 6 pairs of a better and a worse answer <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **D** Only the difference between the two scores matters for the loss <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> 4 choose 2 = 6. With r⁺ = 1.4 and r⁻ = −0.6: −log σ(2.0) ≈ 0.13. Adding a constant to both scores leaves the loss unchanged. During RL the reward model is frozen.

---

## Q17 · RAG retrieval
<small>Applied</small>

**A course assistant can retrieve with BM25 or with embeddings. What is true?**

- [ ] **A** Retrieval changes the model’s weights <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **B** Embeddings can match “Is there a second chance if I fail?” to the Retake entry, with no word in common <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **C** A reranker reads the question and the passage together <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **D** Embeddings reliably tell “Test 1” and “Test 2” apart <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>

> [!success]- Answer (click to open)
> **B, C** · A ✗ · B ✓ · C ✓ · D ✗
>
> Embeddings capture meaning but blur numbers and names; BM25 matches words. A reranker scores each (question, passage) pair jointly: best, but slower, so only on the top few. RAG changes the context, never the weights.

---

## Q18 · Elo
<small>Applied</small>

**Two models in the Arena are rated 1526 and 1504. What is true?**

- [ ] **A** The leader is expected to win about 53 % of comparisons <small>· look in [[2026-11-18 - Applied Generative AI I#Elo (Arena)|Applied · Elo]]</small>
- [ ] **B** The 22-point gap shows the leader is clearly better for your task <small>· look in [[2026-11-18 - Applied Generative AI I#Elo (Arena)|Applied · Elo]]</small>
- [ ] **C** Adding 100 points to both ratings changes the expected result <small>· look in [[2026-11-18 - Applied Generative AI I#Elo (Arena)|Applied · Elo]]</small>
- [ ] **D** A 400-point gap would mean odds of about 10 : 1 <small>· look in [[2026-11-18 - Applied Generative AI I#Elo (Arena)|Applied · Elo]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> E = 1 / (1 + 10^(−22/400)) ≈ 0.53; for 400: 1 / (1 + 10⁻¹) ≈ 0.91. Only the difference matters. A preference vote is not your task: benchmarks narrow the choice, your own eval decides.

---

## Q19 · running locally
<small>Applied</small>

**You want to run open-weight models on your own machine. What is true?**

- [ ] **A** An 8B model at 4 bit needs about 4 GB for its weights <small>· look in [[2026-11-18 - Applied Generative AI I#Running a model locally|Applied · Running a model locally]]</small>
- [ ] **B** In the lecture, Qwen3-1.7B at 4 bit used less memory than Qwen3-0.6B at 16 bit and scored better <small>· look in [[2026-11-18 - Applied Generative AI I#Running a model locally|Applied · Running a model locally]]</small>
- [ ] **C** Decode speed is bounded by memory bandwidth ÷ bytes of active weights <small>· look in [[2026-11-18 - Applied Generative AI I#Running a model locally|Applied · Running a model locally]]</small>
- [ ] **D** Quantising from 16 to 4 bit usually makes decoding slower <small>· look in [[2026-11-18 - Applied Generative AI I#Running a model locally|Applied · Running a model locally]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> memory ≈ parameters × bytes per weight (+ KV cache): 8B × 0.5 B = 4 GB. Each token reads every active weight once, so fewer bytes means faster (11.5 → 25.6 tok/s). 1,359 MB vs 1,509 MB, 86 % vs 84 % pass@1.

---

## Q20 · chat template and tools
<small>Applied</small>

**Your code sends a messages list plus one tool definition to a local Qwen3. What does the model actually get?**

- [ ] **A** The JSON request object, read by a special JSON input layer <small>· look in [[2026-11-18 - Applied Generative AI I#Station 4 · Template and tokens|Applied · Station 4 · Template and tokens]]</small>
- [ ] **B** Nothing about the tool; the server calls it before the model runs <small>· look in [[2026-11-18 - Applied Generative AI I#Station 4 · Template and tokens|Applied · Station 4 · Template and tokens]]</small>
- [ ] **C** The tool’s name, description and parameters as text in the system prompt <small>· look in [[2026-11-18 - Applied Generative AI I#Station 4 · Template and tokens|Applied · Station 4 · Template and tokens]]</small>
- [ ] **D** Text in which <|im_start|> and <|im_end|> mark the messages, each one token <small>· look in [[2026-11-18 - Applied Generative AI I#Station 4 · Template and tokens|Applied · Station 4 · Template and tokens]]</small>

> [!success]- Answer (click to open)
> **C, D** · A ✗ · B ✗ · C ✓ · D ✓
>
> The server applies the chat template; the special tokens were learned in post-training. A tool is just text in the prompt; the model “calls” it by writing <tool_call>…</tool_call>, and your code runs it.

