---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Mock Test B

20 new questions, same format and spread.

> [!info] How the real test is scored (from the Preliminary Discussion)
> 20 questions · 40 minutes · closed book · **true multiple choice: 0 to 4 options may be correct**, and you are not told how many.
> Per question (1.5 points): **+1.5 ÷ n** for each correct tick (n = number of correct options), **−0.75** for each wrong tick, never below 0 per question. If no option is correct, **no tick at all = 1.5**.
> Tick the boxes (they are clickable in Obsidian), then open the collapsed answer. Score yourself with the key at the bottom.


## Q1 · symbolic AI
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**A system uses the rule “if horns and udder and hooves and tail then cow”. What is true?**

- [ ] **A** It classifies a dehorned cow as not a cow
- [ ] **B** It is an example of subsymbolic AI
- [ ] **C** A neural network trained on cow pictures can be fixed the same way, by editing one weight
- [ ] **D** Its mistake can be fixed directly by editing the rule

> [!success]- Answer: A, D
> Symbolic systems reason only within explicitly encoded patterns, so unseen cases break them; but the rules are readable and editable. Subsymbolic models can only be corrected indirectly, by retraining on other data.
> *Introduction to Generative AI · symbolic AI*

## Q2 · sound as tokens
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**Sound is turned into tokens. What is true?**

- [ ] **A** 2 seconds at 24 kHz are 48,000 numbers
- [ ] **B** Residual vector quantisation applies a further codebook to what the previous one left out
- [ ] **C** T seconds at sample rate f_s give a vector x with f_s · T numbers
- [ ] **D** Each audio sample becomes one token

> [!success]- Answer: A, B, C
> A window of samples goes through a learned encoder to one vector zₜ; optionally it is snapped to the nearest codebook entry, often repeated on the remainder (EnCodec). One token per window, not per sample.
> *Introduction to Generative AI · sound as tokens*

## Q3 · contextual embeddings
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**Qwen3-0.6B reads “The cat says meow. The cow says”. What is true about the vector of the last token, ·says?**

- [ ] **A** Only the last layers turn it into a confident prediction of ·moo
- [ ] **B** Already a few layers in, it reads as ·moo with high probability
- [ ] **C** Right after the embedding lookup, it is the same vector as ·says in any other sentence
- [ ] **D** The embedding table is where the context changes it

> [!success]- Answer: A, C
> The lookup is static. Read through the final layer, layer 25 still says ·cow; ·moo appears at layer 26 and reaches 93.5 % at layer 28. The context comes in through attention, not through the table.
> *Introduction to Generative AI · contextual embeddings*

## Q4 · mask without positional encoding
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**A decoder has causal masked attention but no positional encoding. It reads “dog bites man” and “man bites dog” (one token per word). What is true?**

- [ ] **A** The output for the first token depends only on that token
- [ ] **B** The outputs are the same vectors as before, only in permuted order
- [ ] **C** “dog” gets the same output vector in both sentences
- [ ] **D** In principle the model can tell positions apart, since each position sees a different number of tokens

> [!success]- Answer: A, D
> The permutation argument holds only without the mask. With it, “dog” first sees only itself; “dog” last sees all three words. That is why the example test said: without positional encoding and without a mask.
> *Theoretical Foundations of Language Modelling · mask without positional encoding*

## Q5 · softmax
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**One row of scaled scores is s = (2, 1, 0). What is true about the softmax weights?**

- [ ] **A** The largest score gets the largest weight
- [ ] **B** Adding 5 to every score changes the weights
- [ ] **C** They add up to 1
- [ ] **D** Multiplying every score by 10 brings the weights closer to (1, 0, 0)

> [!success]- Answer: A, C, D
> softmax(s)ⱼ = e^sⱼ / Σ e^sᵢ: positive, sums to 1, keeps the order. A constant shift cancels out. Larger scores sharpen it toward one-hot: exactly why scores are divided by √dₖ.
> *Theoretical Foundations of Language Modelling · softmax*

## Q6 · shapes in attention
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**X has shape n × d_model; W^Q and W^K are d_model × dₖ; W^V is d_model × d_v. What is true?**

- [ ] **A** QKᵀ has shape dₖ × dₖ
- [ ] **B** QKᵀ has shape n × n
- [ ] **C** The output softmax(QKᵀ/√dₖ)V has shape n × d_v
- [ ] **D** Row i of the weights says how much token i takes from each token j

> [!success]- Answer: B, C, D
> Q and K are n × dₖ, so QKᵀ is n × n: a score for every pair. Softmax runs along each row (over j). Weights (n × n) times V (n × d_v) give one new vector per token.
> *Theoretical Foundations of Language Modelling · shapes in attention*

## Q7 · positional encodings
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**Learned absolute vs sinusoidal positional encodings. What is true?**

- [ ] **A** With sinusoidal encodings, k positions later is the same fixed turn of the vector from any position
- [ ] **B** A learned table with 1,024 rows has no vector for position 1,024
- [ ] **C** Sinusoidal encodings have no parameters to learn
- [ ] **D** Both are applied by rotating queries and keys

> [!success]- Answer: A, B, C
> GPT-2’s learned table: fixed context length, late rows seen only in long texts. Sinusoidal: computed for any position; distances only implicit. Both are added to the token vectors; rotating q and k is RoPE.
> *Theoretical Foundations of Language Modelling · positional encodings*

## Q8 · output layer
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**After the last block, “The cow says” gives a matrix H of shape 3 × 1,024 in Qwen3-0.6B. What is true?**

- [ ] **A** In Qwen3-0.6B the output layer shares its weights with the embedding table
- [ ] **B** To generate the next token, only the last row is needed
- [ ] **C** A linear layer maps each row to 151,936 logits
- [ ] **D** Softmax turns each row into a distribution over the next token

> [!success]- Answer: A, B, C, D
> Linear d → V gives one number per vocabulary entry; softmax makes them probabilities. Training scores every row (each predicts the token after it); generation uses only the last. Qwen3-0.6B ties the head to the embedding table.
> *Theoretical Foundations of Language Modelling · output layer*

## Q9 · scaling laws
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**What is true about scaling laws?**

- [ ] **A** Loss falls as a power law in compute: a straight line on log–log axes
- [ ] **B** When compute grows, compute-optimal parameters and tokens grow by roughly the same factor
- [ ] **C** Chinchilla: compute-optimal training uses about 20 tokens per parameter
- [ ] **D** The Chinchilla-optimal model also minimises the cost of serving it

> [!success]- Answer: A, B, C
> Kaplan: log-linear trends in C, D, N. Chinchilla: N_opt ∝ C^0.49, D_opt ∝ C^0.51. It optimises training compute only: serving a bigger model costs more, which is why the pilot over-trains a small one.
> *Training Paradigms I · scaling laws*

## Q10 · mixture of experts
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**The pilot activates 1 shared + 8 routed experts out of 256 routed, each 768 wide, with d = 2,048. What is true?**

- [ ] **A** It has about 30B parameters in total but per-token compute like a 2B model
- [ ] **B** The active feed-forward width is 9 · 768 = 6,912, about 3.4 · d
- [ ] **C** The router picks the experts once per document
- [ ] **D** During training every token goes through all 256 experts

> [!success]- Answer: A, B
> Only the selected experts run; the router decides per token (and per layer). Total ÷ active ≈ 32 keeps the knowledge capacity of a much larger model at the compute of the active part.
> *Training Paradigms I · mixture of experts*

## Q11 · the loss
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**The training loss per token is −log p, with p the probability given to the true next token. What is true?**

- [ ] **A** A model that spreads probability evenly over a 151,936-token vocabulary has a loss of about 12
- [ ] **B** p = 0.5 gives a loss of about 0.69
- [ ] **C** The batch loss is the sum of the GPUs’ mean losses
- [ ] **D** p = 1 gives a loss of 0

> [!success]- Answer: A, B, D
> −log 1 = 0, −log 0.5 = ln 2 ≈ 0.69, −log(1/151,936) = ln 151,936 ≈ 11.9. The batch loss is the mean over all tokens; with equal counts per GPU that is the mean of the GPUs’ means, not their sum.
> *Training Paradigms I · the loss*

## Q12 · gradient clipping
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**Gradients are clipped to norm 1.0. What is true?**

- [ ] **A** A gradient of norm 0.5 is left as it is
- [ ] **B** A gradient of norm 8 is scaled down to norm 1, keeping its direction
- [ ] **C** Each weight’s gradient is clipped to [−1, 1] on its own
- [ ] **D** It guards against the huge gradient of a loss spike

> [!success]- Answer: A, B, D
> The norm of the gradient of all weights together is computed; if it exceeds 1.0, the whole vector is rescaled. Usual steps are untouched; a spike step keeps its direction but gets length 1.
> *Training Paradigms I · gradient clipping*

## Q13 · mid-training
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**What is part of mid-training?**

- [ ] **A** Finishing on the best data, including maths and reasoning
- [ ] **B** Extending the context to longer rows
- [ ] **C** Adding thinking traces as chat conversations with user and assistant roles
- [ ] **D** Cooling down the learning rate

> [!success]- Answer: A, B, D
> Mid-training: cool down, finish on the best data, extend the context. Thinking traces go in as plain text without chat format; the chat template comes with post-training.
> *Training Paradigms II · mid-training*

## Q14 · YaRN
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**Olmo 3 7B (RoPE base 500,000) is extended from 8K to 64K with YaRN, scale 8. What is true?**

- [ ] **A** Dividing every pair by 8 instead would blur the order of nearby tokens
- [ ] **B** The fastest RoPE pairs are left unchanged
- [ ] **C** The slowest pairs are slowed down by a factor of 8
- [ ] **D** With YaRN, no training on long documents is needed

> [!success]- Answer: A, B, C
> Pair 1 turns 1,304 times in 8K: every angle seen, keep it. Pair 48 turns only 31°: beyond 8K it would reach unseen angles, so ÷8 maps 0–64K into the seen range. A long-context stage (64K rows) still follows.
> *Training Paradigms II · YaRN*

## Q15 · the baseline
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**Four answers get rewards 0.8, 0.5, 0.3 and 0.1. Compare L = −r · Σ log p with L = −(r − b) · Σ log p, b = 0.425. What is true?**

- [ ] **A** Without the baseline, all four answers are made more likely
- [ ] **B** In PPO, the baseline is the value model’s prediction Vₜ
- [ ] **C** The baseline changes which answer has the highest reward
- [ ] **D** With the baseline, answers below the average are made less likely

> [!success]- Answer: A, B, D
> All rewards are positive, so −r · Σ log p pushes every answer up, the bad ones just less. r − b is negative for 0.3 and 0.1. Subtracting a constant does not reorder rewards. PPO: Aₜ = Gₜ − Vₜ; GRPO: the group mean.
> *Training Paradigms II · the baseline*

## Q16 · reward model
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**A reward model is trained on people’s rankings of 4 answers to one prompt. What is true?**

- [ ] **A** Each pair’s loss is −log σ(r⁺ − r⁻)
- [ ] **B** During RL, the reward model keeps training alongside the policy
- [ ] **C** The ranking gives 6 pairs of a better and a worse answer
- [ ] **D** Only the difference between the two scores matters for the loss

> [!success]- Answer: A, C, D
> 4 choose 2 = 6. With r⁺ = 1.4 and r⁻ = −0.6: −log σ(2.0) ≈ 0.13. Adding a constant to both scores leaves the loss unchanged. During RL the reward model is frozen.
> *Training Paradigms II · reward model*

## Q17 · RAG retrieval
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**A course assistant can retrieve with BM25 or with embeddings. What is true?**

- [ ] **A** Retrieval changes the model’s weights
- [ ] **B** Embeddings can match “Is there a second chance if I fail?” to the Retake entry, with no word in common
- [ ] **C** A reranker reads the question and the passage together
- [ ] **D** Embeddings reliably tell “Test 1” and “Test 2” apart

> [!success]- Answer: B, C
> Embeddings capture meaning but blur numbers and names; BM25 matches words. A reranker scores each (question, passage) pair jointly: best, but slower, so only on the top few. RAG changes the context, never the weights.
> *Applied Generative AI I · RAG retrieval*

## Q18 · Elo
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**Two models in the Arena are rated 1526 and 1504. What is true?**

- [ ] **A** The leader is expected to win about 53 % of comparisons
- [ ] **B** The 22-point gap shows the leader is clearly better for your task
- [ ] **C** Adding 100 points to both ratings changes the expected result
- [ ] **D** A 400-point gap would mean odds of about 10 : 1

> [!success]- Answer: A, D
> E = 1 / (1 + 10^(−22/400)) ≈ 0.53; for 400: 1 / (1 + 10⁻¹) ≈ 0.91. Only the difference matters. A preference vote is not your task: benchmarks narrow the choice, your own eval decides.
> *Applied Generative AI I · Elo*

## Q19 · running locally
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**You want to run open-weight models on your own machine. What is true?**

- [ ] **A** An 8B model at 4 bit needs about 4 GB for its weights
- [ ] **B** In the lecture, Qwen3-1.7B at 4 bit used less memory than Qwen3-0.6B at 16 bit and scored better
- [ ] **C** Decode speed is bounded by memory bandwidth ÷ bytes of active weights
- [ ] **D** Quantising from 16 to 4 bit usually makes decoding slower

> [!success]- Answer: A, B, C
> memory ≈ parameters × bytes per weight (+ KV cache): 8B × 0.5 B = 4 GB. Each token reads every active weight once, so fewer bytes means faster (11.5 → 25.6 tok/s). 1,359 MB vs 1,509 MB, 86 % vs 84 % pass@1.
> *Applied Generative AI I · running locally*

## Q20 · chat template and tools
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**Your code sends a messages list plus one tool definition to a local Qwen3. What does the model actually get?**

- [ ] **A** The JSON request object, read by a special JSON input layer
- [ ] **B** Nothing about the tool; the server calls it before the model runs
- [ ] **C** The tool’s name, description and parameters as text in the system prompt
- [ ] **D** Text in which <|im_start|> and <|im_end|> mark the messages, each one token

> [!success]- Answer: C, D
> The server applies the chat template; the special tokens were learned in post-training. A tool is just text in the prompt; the model “calls” it by writing <tool_call>…</tool_call>, and your code runs it.
> *Applied Generative AI I · chat template and tools*

---

> [!example]- Answer key
>
> | Q | Correct | Lecture |
> |---|---|---|
> | 1 | A, D | Introduction to Generative AI |
> | 2 | A, B, C | Introduction to Generative AI |
> | 3 | A, C | Introduction to Generative AI |
> | 4 | A, D | Theoretical Foundations of Language Modelling |
> | 5 | A, C, D | Theoretical Foundations of Language Modelling |
> | 6 | B, C, D | Theoretical Foundations of Language Modelling |
> | 7 | A, B, C | Theoretical Foundations of Language Modelling |
> | 8 | A, B, C, D | Theoretical Foundations of Language Modelling |
> | 9 | A, B, C | Training Paradigms I |
> | 10 | A, B | Training Paradigms I |
> | 11 | A, B, D | Training Paradigms I |
> | 12 | A, B, D | Training Paradigms I |
> | 13 | A, B, D | Training Paradigms II |
> | 14 | A, B, C | Training Paradigms II |
> | 15 | A, B, D | Training Paradigms II |
> | 16 | A, C, D | Training Paradigms II |
> | 17 | B, C | Applied Generative AI I |
> | 18 | A, D | Applied Generative AI I |
> | 19 | A, B, C | Applied Generative AI I |
> | 20 | C, D | Applied Generative AI I |
