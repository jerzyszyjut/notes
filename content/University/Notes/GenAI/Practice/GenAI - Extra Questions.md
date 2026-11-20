---
publish: true
tags: [genai, practice, test1]
course: Generative AI (194.207)
---

# Extra Questions

Additional questions by topic, for drilling the parts that the mock tests touch less.

> [!info] How the real test is scored (from the Preliminary Discussion)
> 20 questions · 40 minutes · closed book · **true multiple choice: 0 to 4 options may be correct**, and you are not told how many.
> Per question (1.5 points): **+1.5 ÷ n** for each correct tick (n = number of correct options), **−0.75** for each wrong tick, never below 0 per question. If no option is correct, **no tick at all = 1.5**.
> Tick the boxes (they are clickable in Obsidian), then open the collapsed answer. Score yourself with the key at the bottom.


## Q1 · BPE training
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**What is true about training a byte-level BPE tokenizer?**

- [ ] **A** Token IDs that are numerically close have similar meanings
- [ ] **B** Training starts from a vocabulary of the 256 byte values
- [ ] **C** Each merge adds the most frequent adjacent pair as a new token
- [ ] **D** A word never seen in training cannot be tokenized

> [!success]- Answer: B, C
> Start from 256 bytes, repeatedly merge the most frequent pair (256 ·c, 257 he, 258 the …), then add special tokens. Unseen words fall back to smaller pieces. An ID is an address, not a meaning.
> *Introduction to Generative AI · BPE training*

## Q2 · patches
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**A picture is cut into 8 × 8 RGB patches, each embedded as W xᵢ + pᵢ. What is true?**

- [ ] **A** pᵢ tells the model where (row, column) the patch is
- [ ] **B** Snapping to a codebook entry is needed before a transformer can read the patch
- [ ] **C** xᵢ has 192 numbers
- [ ] **D** Each patch position has its own matrix W

> [!success]- Answer: A, C
> 8 · 8 · 3 = 192. One learned W ∈ ℝ^(d×192) is shared by all patches; the position comes from pᵢ. Snapping to a codebook ID is optional; continuous vectors can go straight in.
> *Introduction to Generative AI · patches*

## Q3 · autoregression vs diffusion
<small>[[2026-10-21 - Introduction to Generative AI|Intro]]</small>

**What is true about autoregressive and diffusion generation?**

- [ ] **A** An autoregressive model keeps each token fixed once it is written
- [ ] **B** For text, masks can play the role of noise in diffusion
- [ ] **C** Diffusion updates every piece at every step
- [ ] **D** Autoregressive models cannot generate pictures

> [!success]- Answer: A, B, C
> Pictures can be cut into patches and generated token by token too (e.g. Emu3). Diffusion noise per modality: Gaussian noise for pixels, masks for text, probabilities for DNA.
> *Introduction to Generative AI · autoregression vs diffusion*

## Q4 · multi-head attention
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**The original transformer has d_model = 512 and h = 8 heads. What is true?**

- [ ] **A** Each head works with queries and keys of length 64
- [ ] **B** The heads’ outputs are concatenated and multiplied by W^O
- [ ] **C** 8 heads cost about 8× the compute of one head of width 512
- [ ] **D** Each head sees only one eighth of the tokens

> [!success]- Answer: A, B
> Q, K and V are cut into h slices of d_model / h = 64; every head sees all tokens, from its own perspective. Total width stays 512, so the cost is about that of one wide head.
> *Theoretical Foundations of Language Modelling · multi-head attention*

## Q5 · feed-forward
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**What is true about the feed-forward layer in a decoder block?**

- [ ] **A** In a mixture of experts, the router picks experts once per sequence
- [ ] **B** It applies the same network to each token on its own
- [ ] **C** It holds most of a block’s weights
- [ ] **D** It is where information from other positions is combined

> [!success]- Answer: B, C
> 512 → 2,048 → 512: 2 · 512 · 2,048 ≈ 2.1M weights vs 4 · 512² ≈ 1.05M for attention. It turns what attention gathered into new features (e.g. Eiffel Tower → Paris). MoE routing is per token.
> *Theoretical Foundations of Language Modelling · feed-forward*

## Q6 · values
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**Why does attention use a learned value projection, v = x W^V?**

- [ ] **A** W^V is applied to the attention weights after the softmax
- [ ] **B** Using the embedding itself as the value can also work
- [ ] **C** So that how a token shifts others can differ from where it sits in meaning space
- [ ] **D** Without W^V, the attention weights could not be computed

> [!success]- Answer: B, C
> Weights come from Q and K; values decide what is moved. The simplest shift uses x itself, and that can work (He & Hofmann, ICLR 2024), but a separate projection usually does better. V = X W^V, then weights × V.
> *Theoretical Foundations of Language Modelling · values*

## Q7 · two meanings of bank
<small>[[2026-10-28 - Theoretical Foundations of Language Modelling|LM]]</small>

**“He sat on the river bank” vs “She put her money in the bank”. What is true about ·bank after the last layer of Qwen3-0.6B?**

- [ ] **A** The two vectors differ
- [ ] **B** The difference comes from attention to the other words
- [ ] **C** The embedding lookups already differ
- [ ] **D** The feed-forward layers alone could produce the difference, without attention

> [!success]- Answer: A, B
> Same lookup row in both; after 28 layers cosine similarity 0.79. Only attention brings in the neighbours; the feed-forward sees each token alone.
> *Theoretical Foundations of Language Modelling · two meanings of bank*

## Q8 · width vs depth
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**The pilot has 2B active parameters, d = 2,048, 24 blocks and 16 heads. What is true?**

- [ ] **A** Each head has 128 dimensions
- [ ] **B** d ÷ blocks ≈ 85, inside the good range of 20–250
- [ ] **C** 4,096 × 3 blocks would be too deep
- [ ] **D** Inside that range, loss changes by only about 1 %

> [!success]- Answer: A, B, D
> 2,048 / 24 ≈ 85; 2,048 / 16 = 128. 4,096 × 3 is too wide, 1,024 × 116 too deep (Kaplan Fig. 5, EuroLLM-1.7B’s shape).
> *Training Paradigms I · width vs depth*

## Q9 · tokenizer fertility
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**The same text needs 7.2× as many tokens in Greek as in English with GPT-2’s tokenizer. What is true?**

- [ ] **A** Less Greek text fits into a fixed context window
- [ ] **B** Greek text costs more compute per sentence
- [ ] **C** A tokenizer trained on EU languages reduces the factor
- [ ] **D** Training longer removes the problem

> [!success]- Answer: A, B, C
> Fertility is a property of the tokenizer, fixed before training. More tokens per sentence means more compute, higher API cost, shorter effective context and slower generation.
> *Training Paradigms I · tokenizer fertility*

## Q10 · packing
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**Documents are packed into training rows of 4,096 tokens. What is true?**

- [ ] **A** A document longer than 4,096 tokens is dropped
- [ ] **B** Packing wastes less compute than padding each document to the row length
- [ ] **C** Every document must be exactly 4,096 tokens long
- [ ] **D** Every document is followed by <|endoftext|>

> [!success]- Answer: B, D
> Join all documents, each ended with <|endoftext|> (151643), and cut into rows: 0 % waste vs 44 % with padding. A long document is cut across rows.
> *Training Paradigms I · packing*

## Q11 · optimiser and schedule
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**The update is θ ← θ − ηₜ · Δₜ. What is true?**

- [ ] **A** With AdamW, Δₜ is simply the raw gradient of the current batch
- [ ] **B** In a warmup–stable–decay schedule the learning rate never changes
- [ ] **C** ηₜ comes from the learning-rate schedule, e.g. warmup, plateau, decay
- [ ] **D** AdamW smooths the gradient and sizes the step per weight

> [!success]- Answer: C, D
> Optimiser (Δ): one batch’s gradient is noisy and badly scaled → smooth it (momentum) and size it per weight (AdamW, Muon). Scheduler (η): start gently, go big, end small.
> *Training Paradigms I · optimiser and schedule*

## Q12 · images in the architecture
<small>[[2026-11-04 - Training Paradigms I|TP I]]</small>

**Ways to add images to a language model. What is true?**

- [ ] **A** In the adapter route, text and images learn together from the start
- [ ] **B** A native model without an encoder is the most proven route among frontier models
- [ ] **C** Adding images late in training (50 % in the last fifth) scored best in Kimi K2.5’s comparison
- [ ] **D** An adapter joins a separately trained vision encoder to a finished language model through a projector

> [!success]- Answer: D
> Adapter (Qwen2.5-VL): cheap, but images join a model built for text. Native with encoder (Kimi K3): learn together, one big joint run. Native without encoder (Fuyu-8B): simplest, less proven. Early, a little at a time (10 %) was best on images and on text.
> *Training Paradigms I · images in the architecture*

## Q13 · workflow or agent
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**What is true about workflows and agents?**

- [ ] **A** A fixed workflow is usually the more expensive and harder to test option
- [ ] **B** In an agent, your code chooses the sequence of steps
- [ ] **C** Agent frameworks run a fundamentally different loop than the 15-line agent.py
- [ ] **D** An agent can run alone as long as a machine checks its work with a check it cannot change

> [!success]- Answer: D
> Workflow: your code chooses the path (cheap, fast, testable). Agent: the model chooses. Underneath, the frameworks run the same loop. Machine checks make unattended runs safe.
> *Applied Generative AI I · workflow or agent*

## Q14 · cost
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**Prices are 2 per million input tokens and 10 per million output tokens (USD). One question uses 2,000 input and 300 output tokens. What is true?**

- [ ] **A** Caching 1,500 of the input tokens makes the output cheaper
- [ ] **B** Output tokens are the larger part of the bill here
- [ ] **C** The request costs about 0.7 US cents
- [ ] **D** Without caching, the input costs 3 cents

> [!success]- Answer: C
> 2,000 · 2 / 10⁶ = 0.4 ¢ input, 300 · 10 / 10⁶ = 0.3 ¢ output: 0.7 ¢ in total, input larger. Caching cuts only the input price (to about 0.43 ¢ in total).
> *Applied Generative AI I · cost*

## Q15 · sampling
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**The post-trained Qwen3-0.6B answers the same question several times. What is true?**

- [ ] **A** Asked for a random number from 1 to 9, it picks one number far more often than 11 %
- [ ] **B** Each token is drawn from a distribution, so the wording can differ between runs
- [ ] **C** Correctness evals reliably detect lost diversity
- [ ] **D** Asking for JSON in the prompt guarantees parseable output

> [!success]- Answer: A, B
> “is held” 67 % vs “was” 32 %. Mode dropping: one number 49 % instead of a uniform 11 %; no answer is wrong, so correctness evals miss it. Asked for JSON, it wrapped the answer in ```json fences and json.loads failed.
> *Applied Generative AI I · sampling*

## Q16 · hallucination
<small>[[2026-11-18 - Applied Generative AI I|Applied]]</small>

**Which of these on its own reliably stops a model from inventing facts?**

- [ ] **A** Enforcing a JSON schema on the output
- [ ] **B** Fine-tuning the model once on the course documents
- [ ] **C** Filling the context window with every document available
- [ ] **D** Setting the temperature to 0

> [!success]- Answer: none (no tick = full points)
> None. Temperature 0 and a schema change the form, not what the model knows. Fine-tuning is poor for new facts. More context is not better (lost in the middle, context rot). What helps: grounding and citing, allowing “I don’t know”, self-checks, sampling and comparing, tools, measuring.
> *Applied Generative AI I · hallucination*

---

> [!example]- Answer key
>
> | Q | Correct | Lecture |
> |---|---|---|
> | 1 | B, C | Introduction to Generative AI |
> | 2 | A, C | Introduction to Generative AI |
> | 3 | A, B, C | Introduction to Generative AI |
> | 4 | A, B | Theoretical Foundations of Language Modelling |
> | 5 | B, C | Theoretical Foundations of Language Modelling |
> | 6 | B, C | Theoretical Foundations of Language Modelling |
> | 7 | A, B | Theoretical Foundations of Language Modelling |
> | 8 | A, B, D | Training Paradigms I |
> | 9 | A, B, C | Training Paradigms I |
> | 10 | B, D | Training Paradigms I |
> | 11 | C, D | Training Paradigms I |
> | 12 | D | Training Paradigms I |
> | 13 | D | Applied Generative AI I |
> | 14 | C | Applied Generative AI I |
> | 15 | A, B | Applied Generative AI I |
> | 16 | none | Applied Generative AI I |
