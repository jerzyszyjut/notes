---
publish: true
tags: [genai, test1, interactive]
course: Generative AI (194.207)
---

# GenAI Lab (interactive)

Live versions of the formulas, one tool at a time, and a quiz with all 120 practice questions scored like the real test.

## Quiz

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm?tool=quiz" width="100%" height="900" style="border:1px solid #8884; border-radius:8px;"></iframe>

Sets: Mock Tests A–D, the practice questions from the notes, or all 120 shuffled; filter by lecture. Every option has a ↗ link to the section of the notes that answers it.

---

## The tools

Each tool is also embedded next to its theory in the lecture notes. Address: `genai-lab.htm?tool=<id>`.

| Tool | id | In the notes |
|---|---|---|
| Attention row: √dₖ, mask, softmax | `attention` | [[2026-10-28 - Theoretical Foundations of Language Modelling#Why divide by √dₖ?\|LM · Why divide by √dₖ?]] |
| RoPE dials | `rope` | [[2026-10-28 - Theoretical Foundations of Language Modelling#RoPE in detail\|LM · RoPE in detail]] |
| LayerNorm on one vector | `layernorm` | [[2026-10-28 - Theoretical Foundations of Language Modelling#5 · Add & Norm – keeping it trainable\|LM §5 Add & Norm]] |
| Compute budget C ≈ 6·N·D | `compute` | [[2026-11-04 - Training Paradigms I#1.4 The pilot's numbers, step by step\|TP I §1.4]] |
| KV cache size, GQA, MLA | `kvcache` | [[2026-11-11 - Training Paradigms II#Excursus – attention cost and the KV cache\|TP II · Excursus]] |
| YaRN | `yarn` | [[2026-11-11 - Training Paradigms II#Stretching the positions – YaRN\|TP II · YaRN]] |
| DPO loss | `dpo` | [[2026-11-11 - Training Paradigms II#B2 · DPO – pairs, one better\|TP II §B2]] |
| On-policy distillation grades | `distill` | [[2026-11-11 - Training Paradigms II#B3 · On-policy distillation – a teacher, token by token\|TP II §B3]] |
| Reward model loss, KL | `rm`, `kl` | [[2026-11-11 - Training Paradigms II#B4 · RLHF – a reward model\|TP II §B4]] |
| PPO ratio, clip, returns | `ppo` | [[2026-11-11 - Training Paradigms II#B7 · PPO\|TP II §B7]] |
| GRPO advantages | `grpo` | [[2026-11-11 - Training Paradigms II#B8 · GRPO (DeepSeekMath)\|TP II §B8]] |
| Elo | `elo` | [[2026-11-18 - Applied Generative AI I#Elo (Arena)\|Applied · Elo]] |
| Memory and tokens/s | `memory` | [[2026-11-18 - Applied Generative AI I#Running a model locally\|Applied · Running locally]] |
| Request cost | `cost` | [[2026-11-18 - Applied Generative AI I#Station 7 · Stream back\|Applied · Station 7]] |
| pass@k vs pass^k | `passk` | [[2026-11-18 - Applied Generative AI I#pass@k vs passᵏ\|Applied · pass@k]] |

> [!info] Where the lab lives
> The page is `attachments/genai-lab.htm` in this folder ([[genai-lab.htm|open the whole lab]]). `pn` publishes it, and every frame loads the published copy, so the frames work on the site and in Obsidian (online).
