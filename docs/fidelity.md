**Layman Rewrite: fidelity.md**

### Model Fidelity & Deviations

Before running our experiment, we checked every attention model against the original authors' reference code to ensure we built them correctly. These notes explain where each model comes from and any deliberate changes we made.

#### Vanilla (The Standard Model)

* **Source:** The original 2017 Transformer paper.
* **Role:** This is our baseline. It is the standard, unmodified model we compare everything else against to measure speed and accuracy.
* **Deviations:** We disabled attention dropout (as explained in Design Decisions).

#### Sparse Attention

* **Source:** "Sparse Transformers" by OpenAI (Child et al., 2019).
* **The Idea:** Instead of looking at every single token at once, this model only looks at a specific, limited pattern of data to save time.
* **Deviations:** The original model was built to generate text (looking only at the past), but our task requires looking at the whole sequence at once. We adjusted the pattern to allow this.
* **Important Note:** Our code perfectly mimics the *accuracy* of Sparse Attention, but it isn't optimized for speed yet. We cannot use this model to make claims about training speed, only about its learning capabilities.

#### FlashAttention

* **Source:** The official FlashAttention library.
* **The Bug Fix:** We discovered a bug where padding (blank spaces added to short sequences) forced the system to abandon FlashAttention and use a slower method. We fixed this by completely removing the padding before processing, ensuring we get the true FlashAttention speedup.
* **Deviations:** Disabled dropout.

#### Linear Attention

* **Source:** "Transformers are RNNs" (Katharopoulos et al., 2020).
* **Outcome:** We verified our code line-by-line against the authors' original code and found it perfectly matches their mathematical logic.
* **Parameters:** This model uses the exact same number of parameters as the Vanilla model, making it a great direct comparison.

#### Linformer

* **Source:** "Linformer" (Wang et al., 2020).
* **Bug Fixed (Initialization):** We found and fixed a bug where the model was generating its starting random numbers too widely, which was throwing off its learning.
* **Correction (Size):** We previously thought this model was 3x larger than the Vanilla model. We realized it reuses a specific matrix, meaning it is actually much closer in size to the standard models.
* **Deviations:** We clear out padded data before the model compresses its information, which is necessary for our specific math tasks.

---

**Draft Email to Steven**

Subject: Honours Project Update: Multi-Task Grid Running & Automated Checkpointing

Hi Steven,

I wanted to share a quick update on where we are with the attention mechanism RCT.

**Where We Are**
We’ve officially moved past the cluster load-shedding and wedged GPU issues. I’ve implemented a pre-flight check in the Slurm scripts that automatically detects dead GPUs and requeues the job to a healthy node. The grid is now self-healing and running unattended.

**The Design Space**
We have fully integrated a multi-task pipeline. Instead of running separate experiments, we are crossing the architectures (Vanilla, Sparse, Flash, Linear, Linformer) across both ListOps and CIFAR-10. This pushes our statistical footing to a robust 16 blocks (8 designs x 2 tasks), giving our final non-parametric tests much more weight without losing task-specific insights.

**What's Next**
The first block of 48 runs is currently executing. Once this clears, I'll push the remaining jobs through the cluster queue. After the full 240-run matrix is complete, the automated pipeline will collect the results and run the Friedman/Nemenyi post-hoc analyses.

You can track the live training metrics on Weights & Biases here:
[W&B Pilot Table](https://wandb.ai/1903697-wits-university/attn-rct-pilot/table?nw=nwuser1903697)

The updated codebase is also public on my GitHub:
[KetsoCodes/attn-rct](https://github.com/KetsoCodes/attn-rct)

Best,
Fortune (Ketso)
