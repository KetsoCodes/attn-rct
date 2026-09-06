# Analysis plan 

**Project:** Investigating Attention Mechanism Modifications with Randomised Control Trials
**Student:** Fortune Mnisi (1903697)
**Supervisors:** Dr Steven James, Prof. Benjamin Rosman

This document locks in exactly how we will analyze our data *before* we actually look at the final results. A major critique of AI research is that scientists often tweak their math after seeing the data to make their results look better (p-hacking). By recording our decisions now, we prove our results are honest.

## 1. How We Count Data (Unit of Observation)
We test 8 different architectural designs. We run each of these setups three separate times using different random seeds. However, we do not count these 3 reruns as independent data points; we just average them together to remove lucky initializations (noise). Treating them as separate points would artificially inflate our sample size and make our math look more confident than it should be. Therefore, our true sample size is 8 (N=8).

## 2. The "All or Nothing" Rule (Completeness)
If any model fails or crashes on a specific design, we throw out that entire design for *all* models. If we didn't do this, memory-heavy models that crash on hard tasks would only be judged on the easy tasks they survived, making them look artificially better than they really are. 

## 3. How We Keep Score (Metrics)
*   **Primary Goal:** Accuracy (higher is better). This is the headline metric for the study.
*   **Secondary Goals:** Training Time and Peak Memory (lower is better). 

We hardcode whether "higher" or "lower" is better into the system so a bug can't accidentally declare a slow, memory-heavy model the winner.

## 4. The Gatekeeper Test (Omnibus)
Before we start comparing specific models against each other, we run a broad test to answer one question: *Are these models actually performing differently at all?* 
*   We use a "Permutation test" (which shuffles the data to double-check the math) because standard statistical formulas actually crash when models rank in perfect order, which is exactly what happens with our memory metric. 
*   If this gatekeeper test says the models are all basically the same, we stop entirely. We will not force further comparisons.

## 5. Head-to-Head Comparisons (Post-Hoc)
If the gatekeeper test proves there are real differences, we will strictly compare every model against our chosen baseline: **FlashAttention**. 
*   We chose this specific approach because running 4 focused comparisons (everyone vs. baseline) is mathematically much stronger than running 10 messy comparisons (everyone vs. everyone). 
*   With our small sample size, doing 10 comparisons would water down our statistical power so much that we might not detect any winners at all.

## 6. Honest Diagrams (Critical-Difference)
When we draw charts grouping the models together, those groups will perfectly match our actual head-to-head decisions (from Step 5). We will not use standard default chart formulas if they contradict our official tables.

## 7. Real-World Impact (Effect Sizes)
Because our sample size is small, we might not always get a "statistically significant" result. Regardless of the strict p-values, we will still report the real-world differences in plain terms (e.g., "Model A was 3.3x faster") so readers understand the practical impact.

## 8. No Chasing Results (Stopping Rule)
We will run the analysis on the data we have. We will not keep adding new designs to the cluster just to push a borderline result over the finish line.

## 9. Known Flaws to Admit
We must explicitly report these limitations so we don't mislead anyone:
*   **Too Much Short Data:** We filtered out sequences longer than 2000 tokens, which accidentally threw away 79% of the training data. The models are mostly learning from short sequences, which defeats the point of testing "long-sequence" efficient models. This strictly limits how high our accuracy can go.
*   **Sparse is Faking its Speed:** Our "Sparse" model is mathematically correct (it gets real accuracy scores), but standard PyTorch handles it so poorly that its memory and speed metrics are completely inaccurate. We are measuring PyTorch's flaw, not the model's true efficiency.
*   **Only One Task:** We are only testing on the ListOps task right now. We cannot claim these models are better for *everything* based on just one test.