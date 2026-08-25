# Design decisions

Decisions that are load-bearing for the paired design. **Do not undo these** without
understanding what they protect; each one exists because the alternative confounds the
intervention with something else.

---

## ADR-001 — attention-weight dropout is 0 for every arm

Vanilla and Linformer can drop softmax weights. Kernel-based linear attention has no
attention matrix to drop. The flash kernel implements its own dropout on a different
RNG stream. So any nonzero `attn_dropout` means a *different regulariser per arm* — an
arm-specific confound in a study whose entire claim is that only the mechanism differs.

Dropout is retained in the fixed frame (post-attention residual and FFN), where it is
identical by construction. Set `cfg.attn_dropout > 0` only if you intend to argue for
it in the write-up.

---

## Precision — flash runs bf16, other arms run fp32

FA-2 kernels accept fp16/bf16 only, so precision is not a free choice for that arm. The
dtype is therefore set once in `base.py` and applied to every arm, rather than each arm
choosing for itself.

Steven's decision: treat 16-bit as part of what FlashAttention *is*. **Consequence for
the write-up: report a PACKAGE effect, not a mechanism effect.** A speedup measured this
way cannot be decomposed into IO-awareness versus tensor cores — the two are collinear
by construction.

`resolve_compute_dtype` prefers bf16 over fp16 on capable hardware because bf16 needs no
loss scaling, and the flash kernel supports both.

---

## Paired initialisation — construction order in `AttentionBase`

`AttentionBase.__init__` builds the fused QKV projection and the output projection in a
fixed order, before any subclass parameters exist. Subclasses add their own parameters
after `super().__init__()`, so they consume RNG draws only at the end and cannot shift
the shared init. Under a fixed seed every arm therefore draws *bit-identical* values for
the parameters they have in common.

This is a stronger form of pairing than matching hyperparameters: designs start from the
same weights. Verified by `test_arms_share_initialisation`.

Linformer's E is the one arm-specific parameter created inside a shared code path, so it
uses a **dedicated generator** (`seed + 90210`). If it drew from the global stream,
every parameter created afterwards would shift and the guarantee would break silently.
Do not reorder any of this.

---

## Padding — filter, never truncate; pad to the batch maximum

Batches pad to the longest sequence *in the batch*, not to `max_len`. Every arm consumes
identical batches, so the comparison stays fair and is simply cheaper than padding
everything to 2000.

Sequences over `max_len` are **filtered**. Truncating a ListOps expression deletes
label-determining operators, which silently corrupts the label — a wrong label is worse
than a smaller dataset. See the open length-cap issue in the handover: this currently
discards a large share of the data and biases what remains toward the short end, which
is a limitation to state rather than a bug to fix here.

---

## `require_flash` — fail loudly

The flash arm raises rather than quietly running a different kernel. Passing an
`attn_mask` to SDPA disqualifies the FLASH_ATTENTION backend, so a masked call falls
back to EFFICIENT_ATTENTION *even on sm_80+* — and the pilot's probe reported hardware
capability rather than the kernel that actually ran, so a 3090 run would have been
logged as true FA-2 while the mem-efficient kernel executed.

`measure_backend` therefore *measures* by making a real call, and `require_flash=True`
must be set for every cluster run producing a reportable efficiency number. This has
already caught one sm_75 run.
