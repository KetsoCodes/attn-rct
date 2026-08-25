# Cluster runbook — mscluster

Ordered. Do not skip ahead; each stage's output is the next stage's precondition.

Guidelines this follows (mscluster Community Guidelines, Feb 2024):
- Never run compute on the login node (§2.2.5). Everything below that costs CPU/GPU
  goes through `sbatch`, including the flash-attn install.
- Prefer `sbatch` over interactive `srun` (§6.2) — load shedding kills interactive jobs.
- Start on `stampede`, escalate to `bigbatch` only when needed (§2.2.7).
- MaxTime = 4320 min (72h) on every partition.
- Home dir is 50GB, and sideloading into it is discouraged (§8.3). Data goes to
  `/datasets/fmnisi/`.
- Acknowledge the cluster in the write-up (§2.3) — text at the bottom of this file.

---

## Stage 0 — access

```bash
ssh fmnisi@146.141.21.100
```

If this fails, open ONE ticket to `support@wits-mss.supportsystem.com`, subject
`TWK HPC Query`, from your Wits address. Do not email staff directly (§2.2.1).

Sanity checks once in:

```bash
sinfo -o "%P %a %l %D %G"     # partitions, timelimits, and whether GRES/gpu is configured
squeue -u fmnisi
df -h /home-mscluster/fmnisi
```

Note whether the `%G` column shows `gpu:...`. If it does, every GPU job below needs
`#SBATCH --gres=gpu:1`. If it shows `(null)` — which is what it currently shows — the
partition hands you the whole node and the flag is unnecessary. **Check this before
submitting anything**; it is the single most common reason a job runs on CPU and
silently takes 40x longer.

## Stage 1 — environment

Two jobs, in order. The second must run on `bigbatch`, because flash-attn needs sm_80+.

```bash
cd /home-mscluster/fmnisi/attn_rct
sbatch cluster/00a_base_env.slurm      # miniconda + the attnrct env + torch
sbatch cluster/00b_flash_attn.slurm    # flash-attn from a prebuilt wheel; bigbatch ONLY
squeue -u fmnisi
cat logs/setup_*.out
```

The wheel is matched to the live interpreter by `cluster/find_flash_wheel.py`. bigbatch
has no `nvcc`, so a source build cannot work — if the wheel step fails, fix the wheel
match rather than falling back to building.

## Stage 2 — verify FA-2 actually engages

```bash
sbatch cluster/01_probe.slurm
cat logs/probe_*.out
```

**This is a gate, not a formality.** The output must contain:

```
capability: sm_86
backend: flash-attn library varlen FA-2 (sm_86) -- VERIFIED
path taken: flash_varlen_fa2
```

If it says `sdpa_fallback_NOT_fa2` or reports anything below sm_80, stop. Every
efficiency number for the baseline would be measuring a different kernel. `biggpu`
(RTX 8000, Turing sm_75) cannot run FA-2 either — `bigbatch` is the only option.

## Stage 3 — ListOps data

```bash
sbatch cluster/02_get_data.slurm
```

The released LRA archive returns **403 AccessDenied** (public bucket access revoked,
verified 2026-08-16), so this stage does not download it. It generates ListOps with the
LRA reference generator at commit `cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8`, producing
96k/2k/2k in `/datasets/fmnisi/lra/listops/` with the official `Source`/`Target` schema.

Note the open length-cap issue: the generator filters on tree node count but writes a
rendered string, so `--max_length 2000` yields sequences averaging ~3103 whitespace
tokens. We filter rather than truncate, which keeps labels correct but retains only
~21% of the data, biased toward shorter sequences. Record that figure with any result —
these numbers are not directly comparable to published LRA results.

## Stage 4 — smoke test on stampede

Cheap plumbing check before spending `bigbatch` time; per §2.2.7 this is what
`stampede` is for. It CANNOT run FA-2 — it only proves the env loads, the data reads,
and a checkpoint writes and resumes.

```bash
sbatch cluster/03_smoke.slurm
```

## Stage 5 — the grid

```bash
# 1. build the manifest (cheap; the login node is fine, it just writes a CSV)
python cluster/make_manifest.py --out cluster/runs.csv

# 2. check the size before you submit
wc -l cluster/runs.csv

# 3. submit, throttled to 6 concurrent — the bigbatch QOS limit (§2.2.8)
sbatch --array=1-$(( $(wc -l < cluster/runs.csv) - 1 ))%6 cluster/train_array.slurm
```

The QOS also caps submitted jobs at 48 per user, so submit the array in blocks rather
than all 120 at once.

Monitoring and control:

```bash
squeue -u fmnisi
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
scancel <JOBID>              # whole array
scancel <JOBID>_7            # one task
scancel --user=fmnisi        # everything
```

---

## W&B

Never commit the key. On the login node, once:

```bash
echo 'export WANDB_API_KEY=<your-key>' >> ~/.bashrc_secrets
chmod 600 ~/.bashrc_secrets
echo '[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets' >> ~/.bashrc
```

The job scripts source this and pass it through. If a compute node has no outbound
network, set `WANDB_MODE=offline` in the job script and run `wandb sync logs/wandb/*`
from the login node afterwards — the probe job reports which case you're in.

## Required acknowledgement (§2.3)

> Computations were performed using High Performance Computing infrastructure provided
> by the Mathematical Sciences Support unit at the University of the Witwatersrand,
> Johannesburg.
