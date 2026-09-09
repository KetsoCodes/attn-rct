# MSCluster Execution Guide

Please follow these steps in exact order. Each stage relies on the successful completion of the previous one.

**Core Cluster Rules (Based on Feb 2024 Guidelines):**
*   **No heavy lifting on the login node:** Any task that requires significant CPU or GPU power (including installations like FlashAttention) must be submitted to a compute node using the `sbatch` command.
*   **Use background jobs (`sbatch`):** Avoid running live, interactive sessions (`srun`). If the cluster experiences power cuts (load shedding), interactive jobs will immediately crash.
*   **Choose the right queue:** Always start testing your code on the smaller `stampede` partition. Only move to the `bigbatch` partition for massive, full-scale runs.
*   **Time limits:** Jobs have a strict maximum time limit of 72 hours (4320 minutes) across all partitions.
*   **Storage limits:** Your personal home folder is capped at 50GB. All large datasets must be downloaded and stored in the dedicated data folder: `/datasets/fmnisi/`.
*   **Credit the cluster:** The final research report must include the standard cluster acknowledgment text, which is provided at the bottom of this file.

---

## Stage 0: System Access
