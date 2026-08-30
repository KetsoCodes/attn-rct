# Addendum for `docs/fidelity.md` — sparse auto-stride

Append this to the sparse section. It records one behavioural change; the rest of the
in-place rewrite changes no computed value and needs no fidelity entry.

---

## Deviation: auto stride resolves against `max_len`, not the padded batch length

**Previously.** `sparse_stride = 0` meant auto, and auto was `round(sqrt(seq_len))`
evaluated inside `_attend`, where `seq_len` is the length the *current batch* happened
to pad to. Batches pad to their own maximum, so the stride — and therefore the attention
pattern — changed from batch to batch according to which sequences were drawn together.

**Now.** Auto resolves once, in `__init__`, as `round(sqrt(max_len))`. At `max_len=2000`
that is a stride of 45, fixed for the run.

**Why.** Two reasons, one experimental and one practical.

The experimental reason is the important one. The intervention in this study is the
attention mechanism, and everything else is meant to be held constant. A stride that
depends on batch composition makes the sparse arm's pattern a random variable driven by
the data loader, which is a nuisance variable inside the treatment itself. No other arm
has an analogous property.

The practical reason is that a per-length stride makes the pattern uncacheable. Each
distinct padded length produced a fresh `(H, S, S)` boolean mask — 32 MB at S=2000 — held
in a cache that was never evicted, per layer. Over an epoch of several hundred distinct
batch lengths that is gigabytes of monotonically growing GPU memory.

**Faithfulness.** Child et al. prescribe a stride close to `sqrt(n)` for a model
operating on sequences of length `n`. Under dynamic padding, `n` is ambiguous: the
model's sequence length is `max_len`, while the padded batch length is an artefact of
how examples were grouped. Reading `n` as the model's length is at least as faithful as
reading it as the padding, and it is the reading that keeps the pattern a property of
the architecture rather than of the shuffle.

**Scope.** Only affects runs with `sparse_stride = 0`. An explicit stride behaves as
before. `cluster/make_manifest.py` currently sets `sparse_stride: 0`, so this is the
path the grid takes; setting it explicitly to 45 in `FIXED` would make the resolved
value visible in `runs.csv` and is worth considering.

**Not affected.** The patterns themselves, the forced diagonal, the empty-row rescue,
and the masked-dense implementation strategy are all unchanged, as is the retracted
block-sparsity finding recorded above — nothing here bears on it.
