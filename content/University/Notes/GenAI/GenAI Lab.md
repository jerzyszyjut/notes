---
publish: true
tags: [genai, test1, interactive]
course: Generative AI (194.207)
---

# GenAI Lab (interactive)

One page with live versions of the formulas, for the topics the 5w1h pages don't make interactive, plus all 76 practice questions with the real test's scoring.

<iframe src="https://jerzyszyjut.github.io/notes/university/notes/genai/genai-lab.htm#quiz" width="100%" height="900" style="border:1px solid #8884; border-radius:8px;"></iframe>

*[[genai-lab.htm|Open the lab full screen]]*

| Tab | What you can play with | Used in |
|---|---|---|
| **Quiz** | Mock Tests A/B/C, Extra questions, or all 76 shuffled; filter by lecture; scored like the test (+1.5/n, −0.75, floor 0) with a 2-minutes-per-question clock | [[_GenAI Test 1 - Start Here\|Start Here]] |
| **Attention** | one row of scores → scale by √dₖ → causal mask → softmax; switch scaling off at large dₖ and watch it go one-hot | [[2026-10-28 - Theoretical Foundations of Language Modelling\|LM]] |
| **RoPE & YaRN** | angle dials per pair for any position and base; YaRN table of turns per pair and the resulting slow-down | [[2026-10-28 - Theoretical Foundations of Language Modelling\|LM]], [[2026-11-11 - Training Paradigms II\|TP II]] |
| **LayerNorm** | μ, σ, γ, β on a six-number vector; RMSNorm switch | [[2026-10-28 - Theoretical Foundations of Language Modelling\|LM]] |
| **Compute** | H100-hours × MFU → C; 6ND; Chinchilla N = √(C/120); vocab share 2Vd; d ÷ blocks | [[2026-11-04 - Training Paradigms I\|TP I]] |
| **KV cache** | bytes per token and per context; GQA and MLA savings | [[2026-11-11 - Training Paradigms II\|TP II]] |
| **DPO · RM · KL** | DPO loss from four log-probs; Bradley–Terry loss; KL between two distributions; on-policy distillation grades | [[2026-11-11 - Training Paradigms II\|TP II]] |
| **PPO · GRPO** | group advantages; ratio and clip; the PPO return/value/advantage table | [[2026-11-11 - Training Paradigms II\|TP II]] |
| **Applied** | pass@k vs pass^k chart; Elo; memory and tokens/s; request cost with caching | [[2026-11-18 - Applied Generative AI I\|Applied]] |

Each tab can be opened directly: add `#quiz`, `#attention`, `#rope`, `#layernorm`, `#compute`, `#kvcache`, `#post`, `#rl` or `#applied` to the address. The lecture notes embed the matching tab right next to the theory.

> [!info] Where the lab lives
> - The page is the file `attachments/genai-lab.htm` in this folder. `pn` publishes it next to the notes, and every embedded frame loads it from the published site, so the frames work on the site **and** inside Obsidian (online).
> - Offline: click *open the lab full screen* (opens the local file in the browser).
> - The quiz has no extra questions beyond the 76 from the mock tests; the 7 official example questions are in [[GenAI - Official Example Questions]].
