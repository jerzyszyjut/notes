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

Two separate pools, so the mock tests stay fresh:

| Pool | Where | Questions |
|---|---|---|
| **Practice by lecture** | at the end of each lecture note, before “Exam traps” | 8 per lecture = 40 (incl. the 7 official ones) |
| **Mock tests** | [[GenAI - Mock Test A\|Mock A]] · [[GenAI - Mock Test B\|Mock B]] · [[GenAI - Mock Test C\|Mock C]] · [[GenAI - Mock Test D\|Mock D]] | 4 × 20, all lectures mixed, 40 minutes each; not in the notes |
| Official sheet | [[GenAI - Official Example Questions]] | the course's 7 questions, ticks as published |

Every option (A–D) links to the section of the notes that answers it: treat each option as its own true/false question. All 120 questions are also in the Quiz of the [[GenAI Lab]], with scoring and a timer.

## How this folder is built

| What | Where |
|---|---|
| Lecture notes | `2026-MM-DD - <lecture>.md`, sorted by date; each section ends with its **lecture slides** (collapsed, click to open) and, where there is one, the matching **interactive tool** (one tool per frame); practice questions at the end |
| Slide pictures | `attachments/GenAI-L<lecture>-slajd<NN>.png` (L1 = Intro … L5 = Applied; NN = page in the lecture PDF) |
| Interactive lab | `attachments/genai-lab.htm`, shown in [[GenAI Lab]] and inside the notes |
| Tests | `Practice/`: Mock Tests A–D, official sheet |
| Website | every note has `publish: true`; run `pn` to publish (Quartz copies the slides and the lab with the notes) |

> [!note] Embedded pages
> The 5w1h pages and the lab are `<iframe>`s. They need internet, inside Obsidian too. The tool frames load the **published** lab (`genai-lab.htm?tool=…`), so changes show up after `pn`.

## Suggested plan (Fri 20 → Wed 25 Nov)

| Day | Do |
|---|---|
| Fri 20 | Intro + LM notes with their practice questions; try the attention, RoPE and LayerNorm tools. **Mock Test A.** |
| Sat 21 | Training Paradigms I note + practice; redo the 6ND / Chinchilla / vocab numbers by hand. **Mock Test B.** |
| Sun 22 | Training Paradigms II note + practice; KV cache, DPO, KL, PPO, GRPO tools. **Mock Test C.** |
| Mon 23 | Applied note + practice. Formula Sheet from memory. |
| Tue 24 | **Mock Test D** under time (40 min); for every wrong option open its linked section; reread the "Exam traps" boxes. |

## The whole course so far, in one chain

Data → **tokens** (BPE / patches / windows) → **embeddings** → *N* × [**masked multi-head attention** with **RoPE** → Add & Norm → **feed-forward / MoE** → Add & Norm] → **linear + softmax** = $p(x_t\mid x_{<t})$ → trained with **−log p** on packed rows, $C\approx6ND$ (**pre-training**) → **mid-training** (cool-down, best data, long context: KV cache, GQA/MLA, YaRN) → **post-training** (SFT, DPO, on-policy distillation, RLHF/RLAIF, RLVR with PPO/GRPO) → wrapped in a **system** (context/RAG, API, template, prefill/decode, decoding constraints, agent loop, evals).
