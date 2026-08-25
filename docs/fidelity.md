# Fidelity notes: provenance and deviations per arm

Every arm was checked against the authors' reference implementation before being
treated as reportable. This file holds that record, so the source files can stay
readable. **These findings are settled — do not re-derive them.** Flag it instead if
any code appears to contradict what is written here.

Batch 1 covers `vanilla` and `sparse`; the remaining arms will be added as their files
are reviewed.

---

## vanilla — `attn_rct/attention/vanilla.py`

**Source.** Vaswani et al. (2017), eq. 1. No canonical repo dependency; this is the
textbook form, and torch's own `nn.MultiheadAttention` is a valid cross-check.

**Role.** The *reference* arm, not the baseline (proposal 4.4.2). It is both the naive
implementation whose wall-clock the IO-aware baseline is measured against, and the
ground truth the flash arm's exactness is verified against. It is deliberately the most
obvious possible implementation, because a reference you have to reason about is not a
reference.

**Deviations.**
- Pre-norm placement lives in the frame, not in the arm.
- No attention-weight dropout — ADR-001.
- Padding handled by additive -inf on key positions.

**Complexity.** Materialises the full `(B, H, S, S)` score matrix: O(S²) time and
memory. At S=2000, H=4, B=8 in fp32 that is ~512 MB for the scores alone, before the
softmax copy. This is the cost the study exists to interrogate.

---

## sparse — `attn_rct/attention/sparse.py`

**Sources.** Child et al. 2019 (arXiv:1904.10509), strided and fixed factorised
attention. Beltagy et al. 2020 (arXiv:2004.05150) for sliding-window plus global
tokens. Reference repo: `openai/sparse_attention`.

**The idea.** Restrict each query to a subset of keys instead of all n. Child et al.
observed that a trained dense Transformer already attends sparsely in most layers, so a
predetermined sparse structure need not cost much quality. Subsets of size O(√n) give
O(n√n) complexity. Child et al. factorise across *heads*: rather than one head
attending to a complicated union of positions, different heads take different
components. We implement this by alternating the pattern on head index.

**Deviations.**
- **(a) Non-causal.** Child et al. define patterns for autoregressive generation, where
  a query attends only to earlier positions. LRA ListOps is encoder-style
  classification, so we make the patterns symmetric. This is a real deviation and must
  be stated in the write-up.
- **(b) Masked-dense, not block-sparse.** See below.
- **(c) No attention-weight dropout** — ADR-001.

**Quality-valid, efficiency-invalid.** The patterns are applied as a mask over a
materialised n × n score matrix. The attention *computed* is exactly the paper's, so
quality results are valid — but memory and wall-clock are those of dense attention, so
no efficiency claim may be made from this arm as written. This is a limitation of our
implementation, **not** a property of the method.

### RETRACTED: "Child's sparse pattern gets zero block-sparse benefit"

An earlier note claimed Child et al.'s strided pattern gets no block-sparse benefit,
because a ~√n stride is finer than a 128-wide GPU block so every block catches a hit.
**That claim is wrong as a statement about their method and must not be reported.** It
described a naive masked implementation, not theirs. Verified against
`openai/sparse_attention` (`attention.py`); two errors compounded:

1. **Causality was dropped.** Their `get_attn_mask` builds causal patterns — strided is
   `(q >= k) & ((q - k) % stride == 0)`, local is a lower-triangular band. Symmetrising
   for encoder use, as we do, roughly doubles block density: 50.8% causal becomes 100%
   symmetric at n=2048, stride 32, block 32.

2. **The transpose was missing.** `blocksparse_attention_impl` calls
   `strided_transpose` *before* the kernel — their comment: strided attention is
   implemented on the transposed matrix to give greater block sparsity. The reshape
   swaps the middle axes and flattens back, so the strided pattern becomes *contiguous*
   in the permuted layout. The kernel therefore never sees a dispersed pattern, and the
   output is permuted back afterwards.

Measured at n=2048, `local_attn_ctx`=32, `blocksize`=32 (their own defaults):

| mask | block density | skippable |
|---|---|---|
| strided, as-is | 50.8% | 49.2% |
| strided, after their transpose | 2.3% | 97.7% |

So Child et al. anticipated and solved this in 2019. Their supported block sizes are
{8, 16, 32, 64} — 128, used in the original measurement, is not among them.

Both numbers are pinned by regression tests in
`attn_rct/tests/test_sparse.py::test_child_transpose_recovers_block_sparsity` and
`::test_symmetrising_inflates_block_density`.

> `ideas_notebook.md` still contains the wrong version and needs correcting.

---

## flash — `attn_rct/attention/flash.py`

**Source.** `Dao-AILab/flash-attention` — `flash_attn_varlen_func` and
`flash_attn.bert_padding.{unpad_input, pad_input}`. Fallback path: torch
`F.scaled_dot_product_attention`, which vendors the same FA-2 kernels.

**Outcome.** Library used directly; FA-2 verified on hardware (bigbatch, sm_86,
`flash-attn 2.8.3.post1` varlen). Agreement with the naive reference: **9.862e-04** in
bf16 at n=2000, over real token positions.

**The bug this arm fixes.** The pilot passed a boolean padding mask into
`F.scaled_dot_product_attention`. Torch's FLASH_ATTENTION backend does not accept an
arbitrary `attn_mask` — only `None` or `is_causal` — so supplying one silently falls
back to EFFICIENT_ATTENTION *even on sm_80+*. Combined with a probe that read compute
capability rather than the kernel that ran, every efficiency number for the baseline
would have been wrong with nothing in the logs to say so.

The fix is the one Dao's own library uses for padded batches: **unpad rather than
mask.** Concatenate the real tokens into one ragged sequence, record the cumulative
sequence lengths, and let the kernel iterate per sequence. Padded keys are not masked
out — they are not there at all. Exactly equivalent to -inf masking, and the only route
to a genuine FA-2 call.

**Deviations.** Dropout disabled (ADR-001). Non-causal only.

---

## linear — `attn_rct/attention/linear.py`

**Source.** Katharopoulos et al. 2020 (ICML). Reference implementation:
`idiap/fast-transformers`, `fast_transformers/attention/linear_attention.py` — the
authors' own lab.

**Outcome — verified line-by-line, no deviations found.** The reference computes:

```
KV = einsum("nshd,nshm->nhmd", K, values)
Z  = 1/(einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + eps)
V  = einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
```

Their layout is (batch, seq, head, dim) and ours is (batch, head, seq, dim), so the
subscripts differ while the contractions match. Confirmed identical: eps **added** (not
clamped), eps default 1e-6, feature map `elu(x)+1`, **no** 1/sqrt(d) scaling anywhere,
and the padding mask applied to K only, propagating into both the KV sum and the
normaliser.

- **Epsilon.** The reference ADDS eps before taking the reciprocal. The pilot used
  `clamp(min=1e-6)`, which is subtly different — clamp is a no-op above the threshold
  while addition always shifts. Matched to the reference.
- **Feature map.** `phi(x) = elu(x) + 1` is non-negative but **not** strictly positive:
  below about -30 in fp32, `exp(x) - 1` rounds to -1 and phi underflows to exactly 0. A
  query whose features all underflow would divide by zero, which is why the reference
  adds eps rather than treating phi as positive. Pinned by
  `test_feature_map_is_non_negative` and `test_eps_guards_total_underflow`.
- **Note.** A third-party adaptation (LoFTR) divides values by sequence length to avoid
  fp16 overflow. Not in the reference and not needed while this arm runs fp32; revisit
  if the precision policy changes.

**Deviations.** Non-causal form only — the causal case admits the RNN reformulation that
gives the paper its title, but ListOps is encoder-style classification. No
attention-weight dropout (ADR-001); note there is no attention matrix to drop, which is
precisely why a nonzero value would mean something different for this arm.

**Parameters.** None beyond the shared projections, so this arm is exactly vanilla's
size — a useful reference point when interpreting Linformer's overhead.

---

## linformer — `attn_rct/attention/linformer.py`

**Source.** Wang et al. 2020, arXiv:2006.04768. Reference implementations:
`facebookresearch/fairseq` (`examples/linformer`, `multihead_linear_attention.py`),
`tatp22/linformer-pytorch`.

**Outcome — 1 bug fixed, 2 deviations documented.**

**Sharing: layerwise**, agreed with supervisor 2026-08. Per the paper, a single
projection matrix E used across all layers, all heads, and for both key and value (so
E = F, one matrix total). The paper reports layerwise as the best-performing of its
three schemes, so this is not merely a parameter-count concession.

**Parameter count — corrected.** Layerwise sharing uses ONE matrix for BOTH key and
value, so E costs `k × max_len` = **512,000**, not the 1,024,000 previously reported. At
the design-space floor the arm is **1.14×** the others, not 3×. *Steven was told 3× and
needs correcting.*

**Initialisation — bug fixed.** Changed from `N(0, 1/k)` (the paper's
Johnson–Lindenstrauss construction) to **`xavier_uniform_` with gain 1/√2**, matching
fairseq's `reset_parameters`. At k=256, max_len=2000 the paper's form gives std 0.0625
against xavier's ~0.021 — roughly 3× too wide. The reference implementation is what
produced the paper's reported results.

**Documented deviations from fairseq:**

1. **Order of operations.** fairseq compresses the input and then projects; we project
   first (in `AttentionBase.project_qkv`) and compress after. The two commute for the
   weight term — E acts on the sequence axis, W on the feature axis — but not for the
   bias: `E(xW + b) = ExW + (E·1)b` versus `(Ex)W + b`. So the arms differ in how the
   projection bias is transformed. Restructuring the base class to compress first would
   break the paired-init guarantee, so this is documented rather than removed. Revisit
   if Linformer underperforms.
2. **Padding.** fairseq does not zero padded positions before compressing, and applies
   no key-padding mask to the compressed weights either — consistent with compressed
   positions no longer corresponding to tokens. We DO zero padded keys and values
   first. Deliberate: fairseq's setting is RoBERTa pretraining on packed sequences where
   padding is rare, whereas ListOps has highly variable lengths and substantial padding
   that would otherwise leak into every compressed position.
3. No attention-weight dropout (ADR-001). Non-causal only.
