---
publish: true
tags: [genai, test1, moc]
aliases: [Home, Test 1]
course: Generative AI (194.207)
---

# 194.207 Generative AI · Test 1 preparation

> [!important] Test 1: Wednesday 25 November, 15:00
> 20 questions · 40 minutes · offline, closed book, no AI · 30 points.
> **True multiple choice:** between 0 and all 4 options can be correct, and you are not told how many.
> Per question 1.5 points: **+1.5 ÷ n** per correct tick (n = number of correct options), **−0.75** per wrong tick, never below 0 per question. If nothing is correct: **no tick = 1.5**.
> Passing the course needs ≥ 30 of 60 points across Test 1 + Test 2 (+ take-home ≥ 15, + one discussion round).
> Scope: all lectures before the test (21 Oct – 18 Nov).

> [!tip] Tactics that follow from the scoring
> - A wrong tick costs 0.75; a correct tick gains 1.5/n. So ticking pays off on average when you are more than about **1/3 sure** (if you think only 1 option is correct), **1/2 sure** (2 correct) or **3/5 sure** (3 correct). The floor of 0 per question softens the risk further.
> - "None correct" questions exist: don't force a tick.
> - The example questions test *mechanisms*, not trivia: why √dₖ, what the mask does, what changes per token in the KV cache, where the 6 comes from. Names of 2026 products are "not exam material".

## Lectures

| # | Lecture | 5w1h page | Note |
|---|---|---|---|
| 0 | Preliminary Discussion (14 Oct) | – | format & grading above |
| 1 | Introduction to Generative AI (21 Oct) | ✓ interactive | [[2026-10-21 - Introduction to Generative AI\|Introduction to Generative AI]] |
| 2 | Theoretical Foundations of Language Modelling (28 Oct) | ✓ static | [[2026-10-28 - Theoretical Foundations of Language Modelling\|Theoretical Foundations of Language Modelling]] |
| 3 | Training Paradigms I (4 Nov) | ✓ static | [[2026-11-04 - Training Paradigms I\|Training Paradigms I]] |
| 4 | Training Paradigms II (11 Nov) | ✗ | [[2026-11-11 - Training Paradigms II\|Training Paradigms II]] |
| 5 | Applied Generative AI I (18 Nov) | ✗ | [[2026-11-18 - Applied Generative AI I\|Applied Generative AI I]] |

Reference: [[GenAI - Glossary|Glossary: symbols & variables]] (every letter in every formula) · [[GenAI - Formula Sheet|Formula Sheet]] (with the lectures' worked numbers) · [[GenAI Lab]] (interactive tools + quiz)

## Practice

- **Start with** [[GenAI - Official Example Questions|the 7 official example questions]]: the course's own sheet, ticks as published.
- [[GenAI - Mock Test A|Mock Test A]] · [[GenAI - Mock Test B|Mock Test B]] · [[GenAI - Mock Test C|Mock Test C]]: 20 questions each, same format and difficulty as the example sheet, spread over all five lectures.
- [[GenAI - Extra Questions|Extra Questions]]: 16 more, by topic.
- The same 76 questions, with automatic scoring and a timer: Quiz tab in [[GenAI Lab]].

## How this folder is built

| What | Where |
|---|---|
| Lecture notes | `2026-MM-DD - <lecture>.md`, sorted by date; each section ends with the **lecture slides** that belong to it (open callouts) and, where there is one, the matching **lab tab** embedded live |
| Slide pictures | `attachments/GenAI-L<lecture>-slajd<NN>.png` (L1 = Intro … L5 = Applied; NN = page in the lecture PDF) |
| Interactive lab | `attachments/genai-lab.htm`, shown in [[GenAI Lab]] and inside the notes |
| Tests | `Practice/`: official sheet, Mock Tests A–C, Extra Questions |
| Website | every note has `publish: true`; run `pn` to publish (Quartz copies the slides and the lab with the notes) |

> [!note] Embedded pages
> The 5w1h pages and the lab are `<iframe>`s. They need internet, inside Obsidian too. The lab frames load the **published** copy, so they show up only after the first `pn`.

## Suggested plan (Fri 20 → Wed 25 Nov)

| Day | Do |
|---|---|
| Fri 20 | Intro + LM notes with the 5w1h pages open; Attention, RoPE and LayerNorm tabs in the lab. **Mock Test A.** |
| Sat 21 | Training Paradigms I note + Compute tab; redo the 6ND / Chinchilla / vocab numbers by hand. |
| Sun 22 | Training Paradigms II note; KV cache, DPO · RM · KL, PPO · GRPO tabs. **Mock Test B.** |
| Mon 23 | Applied note + Applied tab. Formula Sheet from memory. |
| Tue 24 | **Mock Test C** under time (40 min), then Extra Questions on your weakest lecture; reread the "Exam traps" boxes at the end of each note. |

## The whole course so far, in one chain

Data → **tokens** (BPE / patches / windows) → **embeddings** → *N* × [**masked multi-head attention** with **RoPE** → Add & Norm → **feed-forward / MoE** → Add & Norm] → **linear + softmax** = $p(x_t\mid x_{<t})$ → trained with **−log p** on packed rows, $C\approx6ND$ (**pre-training**) → **mid-training** (cool-down, best data, long context: KV cache, GQA/MLA, YaRN) → **post-training** (SFT, DPO, on-policy distillation, RLHF/RLAIF, RLVR with PPO/GRPO) → wrapped in a **system** (context/RAG, API, template, prefill/decode, decoding constraints, agent loop, evals).
