---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
source: "Test 1 · Example Questions (published 25 Nov sheet, 8 pages)"
---

# Official example questions (Test 1)

The seven questions the course published "as on the test, each with its answer". The wording below is the sheet's; the answer letters follow from the sheet's explanations. Do these first, under time: 7 questions ≈ 14 minutes.

> [!info] How the real test is scored (from the Preliminary Discussion)
> 20 questions · 40 minutes · closed book · **true multiple choice: 0 to 4 options may be correct**, and you are not told how many.
> Per question (1.5 points): **+1.5 ÷ n** for each correct tick (n = number of correct options), **−0.75** for each wrong tick, never below 0 per question. If no option is correct, **no tick at all = 1.5**.


## Q1 · the lookup is static, context pulls bank apart
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**In a decoder-only transformer, what is true about “·bank” in these two phrases: “the ·river ·bank” and “the ·savings ·bank”?**

- [ ] **A** The embedding layer gives ·bank the same vector in both phrases
- [ ] **B** After attention, ·bank’s vector holds meaning from the words before it
- [ ] **C** After positional encoding, the two ·bank vectors are different
- [ ] **D** The feed-forward layers add meaning from the neighbouring words

> [!success]- Answer: A, B
> The embedding is a lookup: the same token always gets the same row. ·bank is the third token in both, so its position is the same too. Only attention mixes tokens; the feed-forward looks at each token alone.
> *Introduction to Generative AI · the lookup is static, context pulls bank apart*

## Q2 · attention alone: no order
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**Attention without positional encoding and without a mask reads “dog bites man” and “man bites dog”. Each word is one token. What is true?**

- [ ] **A** The word dog gets the same output vector in both
- [ ] **B** Adding the same vector to every embedding tells the two apart
- [ ] **C** Stacking more of these attention layers tells the two apart
- [ ] **D** The token IDs tell the model the order of the words

> [!success]- Answer: A
> Attention only compares pairs of tokens. Change the order of the input and you get the same outputs in the new order, in every layer. A vector added to every token is the same everywhere, and a token ID says which token it is, not where it stands.
> *Theoretical Foundations of Language Modelling · attention alone: no order*

## Q3 · where C ≈ 6 · N · D comes from
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**The compute for training is about C ≈ 6 · N · D. Where does the 6 come from?**

- [ ] **A** Forward pass: each weight does one multiply and one add per token
- [ ] **B** Backward pass: each weight is used twice, for two different gradients
- [ ] **C** Optimizer step: two more operations for each weight and token
- [ ] **D** Attention: two more operations for comparing the tokens

> [!success]- Answer: A, B
> Forward: 2 operations for each weight and token. Backward: 4. Together: 6. The optimizer runs once per step, not once per token, and the formula leaves attention out.
> *Training Paradigms I · where C ≈ 6 · N · D comes from*

## Q4 · the KV cache: prefill and decode
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**A model generates text one token at a time, with a standard KV cache. What happens for each new token?**

- [ ] **A** Only the new token goes through all the layers
- [ ] **B** Its query is compared with the cached keys of all earlier tokens
- [ ] **C** Its key and value are stored in the cache for later tokens
- [ ] **D** The keys of earlier tokens are updated to include the new token

> [!success]- Answer: A, B, C
> Earlier tokens are needed only inside attention, through their keys and values. Because of the mask, a token never sees later tokens, so its key and value never change.
> *Training Paradigms II · the KV cache: prefill and decode*

## Q5 · softmax, and why the scores are scaled
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**Why are the attention scores divided by √dₖ before the softmax?**

- [ ] **A** Without it, large scores give one token almost all the weight
- [ ] **B** Without it, almost no gradient flows back through the softmax
- [ ] **C** Without it, the scores grow with the length of the sequence
- [ ] **D** Without it, the weights in a row do not add up to 1

> [!success]- Answer: A, B
> Scores grow with dₖ, the length of q and k, not with the length of the sequence. With large scores, softmax gives one token nearly everything and the rest almost 0, so there is almost no gradient. Dividing keeps the order of the scores; softmax itself makes the weights add up to 1.
> *Theoretical Foundations of Language Modelling · softmax, and why the scores are scaled*

## Q6 · RLHF: over-optimization, KL, the loop
<small>[[2026-11-11 - Training Paradigms II|TP II]]</small>

**Why does RLHF penalize the policy with β · KL to the reference model?**

- [ ] **A** It stops the policy from optimizing the reward model’s score too far
- [ ] **B** It keeps the policy close to the SFT model it started from
- [ ] **C** It keeps the reward model close to its weights from before RL
- [ ] **D** It rewards the policy for tokens the reference model finds unlikely

> [!success]- Answer: A, B
> If the policy chases the reward model too far, its score keeps rising while the real quality goes up and then down. The KL term pulls the policy back towards the frozen SFT model, harder the further it moves. The reward model does not change during RL.
> *Training Paradigms II · RLHF: over-optimization, KL, the loop*

## Q7 · masked attention: the chain rule inside attention
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**In masked self-attention, the scores for later tokens are set to −∞ before the softmax. What is true?**

- [ ] **A** The weights for later tokens come out as exactly 0
- [ ] **B** The weights in each row still add up to 1
- [ ] **C** Setting those weights to 0 after the softmax would give the same result
- [ ] **D** Because of the mask, training has to go through the tokens one after the other

> [!success]- Answer: A, B
> e^−∞ = 0, so the softmax gives later tokens no weight and shares the whole 1 among the tokens up to here. Zeroing the weights after the softmax leaves rows that add up to less than 1. The mask is what lets training run every position at once: each sees exactly what it would see while generating.
> *Theoretical Foundations of Language Modelling · masked attention: the chain rule inside attention*

---

> [!example]- Answer key
>
> | Q | Correct | Lecture |
> |---|---|---|
> | 1 | A, B | Introduction to Generative AI |
> | 2 | A | Theoretical Foundations of Language Modelling |
> | 3 | A, B | Training Paradigms I |
> | 4 | A, B, C | Training Paradigms II |
> | 5 | A, B | Theoretical Foundations of Language Modelling |
> | 6 | A, B | Training Paradigms II |
> | 7 | A, B | Theoretical Foundations of Language Modelling |

> [!tip] What the sheet tells you about the test
> Every question asks **why a mechanism works** (lookup vs attention, permutation, 2 + 4 FLOPs, what the cache stores, √dₖ, the KL leash, −∞ before softmax). The wrong options are the classic mix-ups: sequence length instead of dₖ, optimizer in the 6, reward model changing during RL, zeroing after softmax.
