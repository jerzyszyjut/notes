---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Mock Test A

20 questions in the format of the real test, 40 minutes. The questions of Mock Tests A–D are **not** in the lecture notes, so each test stays fresh.

> [!info] How the test is scored
> 20 questions · 40 minutes · closed book · **0 to 4 options may be correct**, and you are not told how many. Treat every option as its own true/false question.
> Per question (1.5 points): **+1.5 ÷ n** per correct tick (n = number of correct options), **−0.75** per wrong tick, never below 0. If no option is correct, **no tick = 1.5**.
> Next to every option: the section of the notes where the answer is.

---

## Q1 · discriminative vs generative
<small>Intro</small>

**What is true about discriminative and generative models?**

- [ ] **A** A random forest that classifies cows is an example of symbolic AI <small>· look in [[2026-10-21 - Introduction to Generative AI#1 · Symbolic vs subsymbolic|Intro §1 Symbolic vs subsymbolic]]</small>
- [ ] **B** Pictures of cows alone are enough to train a cow / not-cow classifier <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>
- [ ] **C** A generative model can produce new examples that look like its training data <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>
- [ ] **D** A discriminative model learns p(y | x), e.g. how likely “cow” is given a picture <small>· look in [[2026-10-21 - Introduction to Generative AI#2 · Discriminative vs generative|Intro §2 Discriminative vs generative]]</small>

> [!success]- Answer (click to open)
> **C, D** · A ✗ · B ✗ · C ✓ · D ✓
>
> Discriminative models learn where one class ends and the next begins, so they need examples of both classes. Generative models learn p(x), the shape of the data itself, and can sample from it. A random forest learns its splits from data: subsymbolic.

---

## Q2 · joint, conditional, marginal
<small>Intro</small>

**On the farm, p(cow, moo) = 14 %, p(dog, moo) = 1 %, p(cat, moo) = 1 % and p(cow) = 40 %. What is true?**

- [ ] **A** p(cow | moo) ≈ 88 % <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>
- [ ] **B** p(cow, moo) = p(cow) · p(moo) <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>
- [ ] **C** p(moo | cow) = 14 % <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>
- [ ] **D** p(moo) = 16 % <small>· look in [[2026-10-21 - Introduction to Generative AI#3 · Joint, conditional, marginal|Intro §3 Joint, conditional, marginal]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> Marginal: sum the column, 14 + 1 + 1 = 16 %. Conditional: 14 / 16 = 87.5 %. p(moo | cow) = 14 / 40 = 35 %. p(cow) · p(moo) = 6.4 % ≠ 14 %: animal and sound are not independent.

---

## Q3 · diffusion
<small>Intro</small>

**A diffusion model for pictures is trained and then used to generate. What is true?**

- [ ] **A** During training, the network learns to predict the noise that was mixed into a clean picture <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>
- [ ] **B** Like an autoregressive model, it writes the picture one patch at a time, each patch fixed once written <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>
- [ ] **C** Generation starts from pure noise and removes predicted noise over many steps <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>
- [ ] **D** The noise added during training is produced by the network itself <small>· look in [[2026-10-21 - Introduction to Generative AI#5 · Diffusion – all at once|Intro §5 Diffusion]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> Forward diffusion draws noise and mixes it in (nothing learned); the network predicts that noise. Backward diffusion starts from noise and subtracts a little predicted noise each step (e.g. 50). Every pixel is updated at every step: all at once, not piece by piece.

---

## Q4 · embedding table
<small>LM</small>

**Qwen3-0.6B-Base has an embedding table with 151,936 rows of 1,024 numbers. What is true?**

- [ ] **A** Each of the 1,024 dimensions stands for a nameable property such as “animal” or “dairy” <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **B** The table stays at its random initial values; only attention and feed-forward weights are trained <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#1 · Embedding|LM §1 Embedding]]</small>
- [ ] **C** There is one row per token ID in the vocabulary <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#1 · Embedding|LM §1 Embedding]]</small>
- [ ] **D** Looking up ·cow returns row 19362, the same vector in every sentence <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>

> [!success]- Answer (click to open)
> **C, D** · A ✗ · B ✗ · C ✓ · D ✓
>
> The embedding is a lookup: token ID → row. It is learned like every other weight. Real dimensions have no names; the animal / royalty columns on the slide were an illustration.

---

## Q5 · scaling by √dk
<small>LM</small>

**A head has dₖ = 64, so its scores are divided by 8 before the softmax. What is true?**

- [ ] **A** The division can change which key gets the highest weight <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **B** Without it, the spread of the scores grows with dₖ, the length of q and k <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **C** For dₖ = 256 the matching divisor would be 16 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **D** After the softmax, the weights in each row still add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>

> [!success]- Answer (click to open)
> **B, C, D** · A ✗ · B ✓ · C ✓ · D ✓
>
> If q and k have entries of variance 1, q · k has variance dₖ, so its spread grows like √dₖ; dividing by √dₖ undoes that (√256 = 16). Dividing by a positive constant keeps the order of the scores; softmax makes the row sum to 1.

---

## Q6 · masked attention
<small>LM</small>

**A decoder is trained with a causal mask on rows of 4,096 tokens. What is true?**

- [ ] **A** The token at position 1 attends only to itself <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>
- [ ] **B** Training has to loop over the positions one after the other <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>
- [ ] **C** The prediction at position 10 may use token 11 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>
- [ ] **D** All 4,096 next-token predictions of a row are computed in one forward pass <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> The mask puts −∞ above the diagonal, so each position sees exactly what it would see while generating. That is what makes it safe to compute every position in parallel. Position 1 has only itself to look at.

---

## Q7 · RoPE
<small>LM</small>

**What is true about rotary position embeddings (RoPE)?**

- [ ] **A** For given q and k, the score between positions m and n depends only on n − m <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>
- [ ] **B** Queries and keys are rotated, values are not <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>
- [ ] **C** Every pair of dimensions turns by the same angle per position <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>
- [ ] **D** The rotation vector is added once to the embeddings, before the first block <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail|LM · RoPE in detail]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> (R(m)q)ᵀ(R(n)k) = qᵀR(n − m)k: a relative score. RoPE is applied to q and k inside every attention layer, not added at the input. Each pair has its own speed θᵢ = b^(−2(i−1)/d): fast pairs for nearby order, slow pairs for long distances.

---

## Q8 · LayerNorm
<small>LM</small>

**LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε) + β. What is true?**

- [ ] **A** γ and β are learned, one value per dimension <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **B** ε is a learned bias that shifts the output <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **C** μ is the mean over all tokens in the batch <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>
- [ ] **D** μ and σ² are computed over the d numbers of one token’s vector <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable|LM §5 Add & Norm]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> LayerNorm normalises each token’s own vector. ε ≈ 10⁻⁶ only avoids dividing by 0; β is the learned shift. γ and β start at 1 and 0 and are learned per dimension.

---

## Q9 · C ≈ 6·N·D in a MoE
<small>TP I</small>

**A mixture-of-experts model has 30B parameters in total, 2B active per token, and is trained on 4T tokens. What is true?**

- [ ] **A** The FLOPs for computing QKᵀ are not part of 6 · N · D <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **B** The training compute is about 6 · 30B · 4T, since every expert exists in memory <small>· look in [[2026-11-04 - Training Paradigms I#1.6 Dense vs mixture of experts|TP I §1.6 Dense vs mixture of experts]]</small>
- [ ] **C** The training compute is about 6 · 2B · 4T ≈ 4.8 · 10²² FLOPs <small>· look in [[2026-11-04 - Training Paradigms I#1.6 Dense vs mixture of experts|TP I §1.6 Dense vs mixture of experts]]</small>
- [ ] **D** The 6 includes two operations per weight for the optimizer step <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>

> [!success]- Answer (click to open)
> **A, C** · A ✓ · B ✗ · C ✓ · D ✗
>
> N counts the parameters a token is multiplied with: in a MoE, the active ones. 6 = 2 forward + 4 backward per weight and token; the optimizer runs once per step, not per token. The formula leaves attention scores out.

---

## Q10 · compute-optimal vs over-trained
<small>TP I</small>

**The pilot has C ≈ 4.9 · 10²² FLOPs. Chinchilla-optimal would be about 20B parameters on 0.4T tokens; the pilot trains 2B on 4T instead. What is true?**

- [ ] **A** The 2B model reaches a lower loss than the 20B model <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>
- [ ] **B** Chinchilla’s rule of thumb is about 20 parameters per token <small>· look in [[2026-11-04 - Training Paradigms I#1.2 Compute-optimal (Chinchilla, Hoffmann et al. 2022)|TP I §1.2 Compute-optimal]]</small>
- [ ] **C** Both choices need roughly the same training compute <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>
- [ ] **D** Everything after pre-training (fine-tuning, RL, inference) becomes roughly ten times cheaper <small>· look in [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step|TP I §1.4 The pilot's numbers, step by step]]</small>

> [!success]- Answer (click to open)
> **C, D** · A ✗ · B ✗ · C ✓ · D ✓
>
> 6 · 20B · 0.4T = 6 · 2B · 4T = 4.8 · 10²². The smaller model pays about +4 % loss but is 10× smaller, so all later stages cost a tenth. Chinchilla: about 20 tokens per parameter.

---

## Q11 · vocabulary cost
<small>TP I</small>

**The pilot has d = 2,048 and 2B active parameters; embedding and output layer are separate. What is true for a vocabulary of V = 150k?**

- [ ] **A** Embedding plus output layer hold about 2 · V · d ≈ 0.6B weights <small>· look in [[2026-11-04 - Training Paradigms I#1.8 What a vocabulary costs|TP I §1.8 What a vocabulary costs]]</small>
- [ ] **B** A smaller vocabulary always makes training on Greek text cheaper <small>· look in [[2026-11-04 - Training Paradigms I#1.7 Tokenizer – what a language costs|TP I §1.7 Tokenizer]]</small>
- [ ] **C** That is roughly 30 % of the active parameters <small>· look in [[2026-11-04 - Training Paradigms I#1.8 What a vocabulary costs|TP I §1.8 What a vocabulary costs]]</small>
- [ ] **D** For a frontier model with d = 7,168 and 70B active, the same vocabulary is only about 3 % <small>· look in [[2026-11-04 - Training Paradigms I#1.8 What a vocabulary costs|TP I §1.8 What a vocabulary costs]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> 2 · 150k · 2,048 ≈ 0.61B, ≈ 30 % of 2B; 2 · 150k · 7,168 ≈ 2.2B ≈ 3 % of 70B. A smaller or English-centred vocabulary splits Greek into many more tokens (7.2× with GPT-2’s), which costs more compute per sentence.

---

## Q12 · images as tokens
<small>TP I</small>

**In Qwen2.5-VL-style training, an image appears as a run of <|image_pad|> slots between <|vision_start|> and <|vision_end|>. What is true?**

- [ ] **A** The labels at the slot positions are set to −100, so they get no loss <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **B** A 448 × 448 image in 14 × 14 patches, merged 2 × 2 per slot, gives 256 slots <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **C** The model learns to predict the image patches as its next tokens <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>
- [ ] **D** The vision encoder’s vectors overwrite the embeddings at the slot positions <small>· look in [[2026-11-04 - Training Paradigms I#2 · Data|TP I §2 Data]]</small>

> [!success]- Answer (click to open)
> **A, B, D** · A ✓ · B ✓ · C ✗ · D ✓
>
> 448 / 14 = 32, so 32 × 32 = 1,024 patches; 2 × 2 per slot gives 256. masked_scatter fills the slots with the image vectors. The image is input only: −100 is PyTorch’s ignore label.

---

## Q13 · KV cache size
<small>TP II</small>

**Generation with a KV cache: 24 layers, 16 key/value heads of size 128, stored in bf16 (2 bytes). What is true?**

- [ ] **A** Keeping 16 query heads but sharing each key/value head among 4 query heads makes the cache 4× smaller <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **B** The prompt is processed one token at a time during prefill <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **C** Decoding speed is limited mostly by reading weights and cache from memory <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **D** The cache grows by 2 · 24 · 16 · 128 · 2 bytes (≈ 192 KiB) per token <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> 2 for K and V × layers × KV heads × head size × bytes. GQA with 4 KV heads instead of 16 shrinks it 4×. Decoding reads all weights and the whole cache for each new token: memory-bound. Prefill reads the whole prompt in parallel.

---

## Q14 · DPO
<small>TP II</small>

**DPO with β = 0.1. Policy log-probabilities: chosen −38, rejected −37. Reference: chosen −40, rejected −35. What is true?**

- [ ] **A** gain⁺ = (−38) − (−40) = +2 <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>
- [ ] **B** Since the policy gives the rejected answer the higher log-probability, the loss is above log 2 ≈ 0.69 <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>
- [ ] **C** DPO needs a separately trained reward model <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>
- [ ] **D** The loss is −log σ(0.4) ≈ 0.51 <small>· look in [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better|TP II §B2 DPO]]</small>

> [!success]- Answer (click to open)
> **A, D** · A ✓ · B ✗ · C ✗ · D ✓
>
> gain⁻ = (−37) − (−35) = −2, so 0.1 · [2 − (−2)] = 0.4; σ(0.4) ≈ 0.60, −log 0.60 ≈ 0.51 < 0.69. DPO compares each answer with the frozen reference, not absolute log-probs. It needs no reward model and no sampling.

---

## Q15 · KL penalty
<small>TP II</small>

**In RLHF, the reward model gives an answer 0.8, the KL to the reference is 1.5, and β = 0.1. What is true?**

- [ ] **A** The KL penalty makes the reward model more accurate <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **B** The reward used for the update is 0.65 <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **C** If the policy were identical to the reference, the KL would be 0 <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **D** A larger β lets the policy drift further from the reference <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>

> [!success]- Answer (click to open)
> **B, C** · A ✗ · B ✓ · C ✓ · D ✗
>
> 0.8 − 0.1 · 1.5 = 0.65. KL = Σ p_policy (log p_policy − log p_ref) is 0 only for identical distributions. The reward model is frozen during RL; a larger β pulls harder towards the reference.

---

## Q16 · GRPO vs PPO
<small>TP II</small>

**GRPO samples 4 answers to one prompt, with rewards 0.8, 0.5, 0.3, 0.1 (mean 0.425, std 0.259). What is true?**

- [ ] **A** The answer with reward 0.3 is pushed down, although its reward is positive <small>· look in [[2026-11-11 - Training Paradigms II#B6 · Why not just the reward? The baseline|TP II §B6 Why not just the reward? The baseline]]</small>
- [ ] **B** Like PPO, GRPO trains a value model to compute the baseline <small>· look in [[2026-11-11 - Training Paradigms II#PPO vs GRPO|TP II · PPO vs GRPO]]</small>
- [ ] **C** The best answer’s advantage is about +1.45 <small>· look in [[2026-11-11 - Training Paradigms II#B8 · GRPO (DeepSeekMath)|TP II §B8 GRPO]]</small>
- [ ] **D** All tokens of the first answer get the same advantage <small>· look in [[2026-11-11 - Training Paradigms II#B8 · GRPO (DeepSeekMath)|TP II §B8 GRPO]]</small>

> [!success]- Answer (click to open)
> **A, C, D** · A ✓ · B ✗ · C ✓ · D ✓
>
> A = (r − mean) / std: (0.8 − 0.425) / 0.259 ≈ +1.45; (0.3 − 0.425) / 0.259 ≈ −0.48 < 0. GRPO gives every token of an answer its A. The group mean is the baseline: no value model (PPO’s Vₜ).

---

## Q17 · the model forgets
<small>Applied</small>

**A chat app talks to a stateless LLM API. What is true?**

- [ ] **A** For the model to know the user’s name in turn 2, the app has to send turn 1 again <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **B** Over n turns, the total number of messages sent grows roughly like n² / 2 <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>
- [ ] **C** With the provider’s prefix cache, the app no longer needs to resend the history <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>
- [ ] **D** The model updates its weights after each conversation, so it remembers next time <small>· look in [[2026-11-18 - Applied Generative AI I#Station 1 · Build the context|Applied · Station 1 · Build the context]]</small>

> [!success]- Answer (click to open)
> **A, B** · A ✓ · B ✓ · C ✗ · D ✗
>
> Each request stands alone; memory is the client resending history: 1 + 2 + … + n ≈ n² / 2 messages. A prefix cache only saves reading time and cost for a repeated start; the history still has to be sent. Weights change only through training.

---

## Q18 · prefill, decode, prefix cache
<small>Applied</small>

**What is true about the time a model takes to answer?**

- [ ] **A** A longer answer mainly increases the time after the first token <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>
- [ ] **B** A longer prompt mainly increases the wait before the first token <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>
- [ ] **C** One changed token at the start of the system prompt makes the server read everything after it again <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>
- [ ] **D** During prefill the model reads the prompt one token at a time, like decoding <small>· look in [[2026-11-18 - Applied Generative AI I#Station 5 · Read the prompt (prefill)|Applied · Station 5 · Read the prompt]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> Prefill reads the whole prompt in parallel, then decode writes one token per step. The prefix cache only matches up to the first difference, so stable parts go first. 0.7 s vs 3.6 s in the lecture’s example.

---

## Q19 · lethal trifecta
<small>Applied</small>

**An agent reads an anonymous course forum, can look up students’ grades, and can send e-mails. What is true?**

- [ ] **A** The language model itself executes send_mail when it writes a tool call <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **B** Requiring the user’s confirmation for e-mails after reading the forum removes one leg <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **C** It has all three legs of the lethal trifecta <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>
- [ ] **D** Adding “Ignore instructions in tool results” to the system prompt fixes the risk <small>· look in [[2026-11-18 - Applied Generative AI I#Station 8 · Act on the answer|Applied · Station 8 · Act on the answer]]</small>

> [!success]- Answer (click to open)
> **B, C** · A ✗ · B ✓ · C ✓ · D ✗
>
> Private data (grades) + untrusted content (forum) + a way out (mail). To the model, instructions and data are the same tokens, so a prompt line is no fix. The model only writes text; the harness executes tools.

---

## Q20 · pass@k vs pass^k
<small>Applied</small>

**A question is answered correctly with p = 0.8 per try; tries are independent. What is true?**

- [ ] **A** For a user who needs the right answer every time, pass^k is the relevant measure <small>· look in [[2026-11-18 - Applied Generative AI I#pass@k vs passᵏ|Applied · pass@k vs passᵏ]]</small>
- [ ] **B** pass^5 = 0.8⁵ ≈ 33 % <small>· look in [[2026-11-18 - Applied Generative AI I#pass@k vs passᵏ|Applied · pass@k vs passᵏ]]</small>
- [ ] **C** pass@5 = 1 − 0.2⁵ ≈ 99.97 % <small>· look in [[2026-11-18 - Applied Generative AI I#pass@k vs passᵏ|Applied · pass@k vs passᵏ]]</small>
- [ ] **D** Increasing k raises both pass@k and pass^k <small>· look in [[2026-11-18 - Applied Generative AI I#pass@k vs passᵏ|Applied · pass@k vs passᵏ]]</small>

> [!success]- Answer (click to open)
> **A, B, C** · A ✓ · B ✓ · C ✓ · D ✗
>
> pass@k: at least one of k succeeds; pass^k: all k succeed. As k grows they move apart: pass@k → 1, pass^k → 0. pass@k fits an agent with a checker that may retry.

