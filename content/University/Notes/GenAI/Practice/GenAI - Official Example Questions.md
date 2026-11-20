---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
source: "Test 1 · Example Questions (course sheet, 8 pages)"
---

# Official example questions (Test 1)

The seven questions the course published “as on the test, each with its answer”. Wording and ticked answers exactly as on the sheet. The same seven also appear in the practice sections of the lecture notes.

> [!info] How the test is scored
> 20 questions · 40 minutes · closed book · **0 to 4 options may be correct**, and you are not told how many. Treat every option as its own true/false question.
> Per question (1.5 points): **+1.5 ÷ n** per correct tick (n = number of correct options), **−0.75** per wrong tick, never below 0. If no option is correct, **no tick = 1.5**.
> Next to every option: the section of the notes where the answer is.

---

## Q1 · the lookup is static, context pulls bank apart
<small>Intro</small>

**In a decoder-only transformer, what is true about “·bank” in these two phrases: “the ·river ·bank” and “the ·savings ·bank”?**

- [ ] **A** The embedding layer gives ·bank the same vector in both phrases <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **B** After attention, ·bank’s vector holds meaning from the words before it <small>· look in [[2026-10-21 - Introduction to Generative AI#7 · Embeddings (start of Language modelling)|Intro §7 Embeddings]]</small>
- [ ] **C** After positional encoding, the two ·bank vectors are different <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **D** The feed-forward layers add meaning from the neighbouring words <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#4 · Feed-forward – knowledge|LM §4 Feed-forward]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> The embedding is a lookup: the same token always gets the same row. ·bank is the third token in both, so its position is the same too. Only attention mixes tokens; the feed-forward looks at each token alone.

---

## Q2 · attention alone: no order
<small>LM</small>

**Attention without positional encoding and without a mask reads “dog bites man” and “man bites dog”. Each word is one token. What is true?**

- [ ] **A** The word dog gets the same output vector in both <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **B** Adding the same vector to every embedding tells the two apart <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **C** Stacking more of these attention layers tells the two apart <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>
- [ ] **D** The token IDs tell the model the order of the words <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#3 · Position – attention alone has no order|LM §3 Position]]</small>

> [!success]- Answer: A
> A ✓ · B ✗ · C ✗ · D ✗
>
> Attention only compares pairs of tokens. Change the order of the input and you get the same outputs in the new order, in every layer. A vector added to every token is the same everywhere, and a token ID says which token it is, not where it stands.

---

## Q3 · where C ≈ 6 · N · D comes from
<small>TP I</small>

**The compute for training is about C ≈ 6 · N · D. Where does the 6 come from?**

- [ ] **A** Forward pass: each weight does one multiply and one add per token <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **B** Backward pass: each weight is used twice, for two different gradients <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **C** Optimizer step: two more operations for each weight and token <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>
- [ ] **D** Attention: two more operations for comparing the tokens <small>· look in [[2026-11-04 - Training Paradigms I#1.3 Where C ≈ 6·N·D comes from|TP I §1.3 Where C ≈ 6·N·D comes from]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> Forward: 2 operations for each weight and token. Backward: 4. Together: 6. The optimizer runs once per step, not once per token, and the formula leaves attention out.

---

## Q4 · the KV cache: prefill and decode
<small>TP II</small>

**A model generates text one token at a time, with a standard KV cache. What happens for each new token?**

- [ ] **A** Only the new token goes through all the layers <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **B** Its query is compared with the cached keys of all earlier tokens <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **C** Its key and value are stored in the cache for later tokens <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>
- [ ] **D** The keys of earlier tokens are updated to include the new token <small>· look in [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache|TP II · Excursus]]</small>

> [!success]- Answer: A, B, C
> A ✓ · B ✓ · C ✓ · D ✗
>
> Earlier tokens are needed only inside attention, through their keys and values. Because of the mask, a token never sees later tokens, so its key and value never change.

---

## Q5 · softmax, and why the scores are scaled
<small>LM</small>

**Why are the attention scores divided by √dₖ before the softmax?**

- [ ] **A** Without it, large scores give one token almost all the weight <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **B** Without it, almost no gradient flows back through the softmax <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **C** Without it, the scores grow with the length of the sequence <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?|LM · Why divide by √dₖ?]]</small>
- [ ] **D** Without it, the weights in a row do not add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> Scores grow with dₖ, the length of q and k, not with the length of the sequence. With large scores, softmax gives one token nearly everything and the rest almost 0, so there is almost no gradient. Dividing keeps the order of the scores; softmax itself makes the weights add up to 1.

---

## Q6 · RLHF: over-optimization, KL, the loop
<small>TP II</small>

**Why does RLHF penalize the policy with β · KL to the reference model?**

- [ ] **A** It stops the policy from optimizing the reward model’s score too far <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **B** It keeps the policy close to the SFT model it started from <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **C** It keeps the reward model close to its weights from before RL <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>
- [ ] **D** It rewards the policy for tokens the reference model finds unlikely <small>· look in [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model|TP II §B4 RLHF]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> If the policy chases the reward model too far, its score keeps rising while the real quality goes up and then down. The KL term pulls the policy back towards the frozen SFT model, harder the further it moves. The reward model does not change during RL.

---

## Q7 · masked attention: the chain rule inside attention
<small>LM</small>

**In masked self-attention, the scores for later tokens are set to −∞ before the softmax. What is true?**

- [ ] **A** The weights for later tokens come out as exactly 0 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **B** The weights in each row still add up to 1 <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **C** Setting those weights to 0 after the softmax would give the same result <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Softmax|LM · Softmax]]</small>
- [ ] **D** Because of the mask, training has to go through the tokens one after the other <small>· look in [[2026-10-28 - Theoretical Foundations of Language Modelling#Masked (causal) attention|LM · Masked]]</small>

> [!success]- Answer: A, B
> A ✓ · B ✓ · C ✗ · D ✗
>
> e^−∞ = 0, so the softmax gives later tokens no weight and shares the whole 1 among the tokens up to here. Zeroing the weights after the softmax leaves rows that add up to less than 1. The mask is what lets training run every position at once: each sees exactly what it would see while generating.

---

> [!tip] What the sheet tells you about the test
> Every question asks **why a mechanism works** (lookup vs attention, permutation, 2 + 4 FLOPs, what the cache stores, √dₖ, the KL leash, −∞ before softmax). The wrong options are the classic mix-ups: sequence length instead of dₖ, optimizer in the 6, reward model changing during RL, zeroing after softmax.
