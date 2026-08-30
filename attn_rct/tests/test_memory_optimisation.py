"""Tests for the in-place scoring path.

The arms were rewritten to stop allocating attention-sized tensors they did not need.
That is a memory change, not a mathematical one, so what these tests establish is that
the rewritten arms compute the same function -- forward and backward -- as the
implementations they replaced. The previous `_attend` bodies are reproduced here
verbatim as reference arms so the comparison is against the real thing rather than
against a paraphrase of it.

Also covered: that the score buffer really is reused, and that the sparse pattern cache
no longer grows with the number of distinct padded batch lengths.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from attn_rct.attention import LinformerAttention, SparseAttention, VanillaAttention
from attn_rct.attention.base import AttentionBase, softmax_
from attn_rct.attention.sparse import PATTERNS


def make_cfg(**overrides):
    """Builds a configuration namespace covering every arm under test."""
    cfg = SimpleNamespace(
        d_model=64, n_heads=4, attention="vanilla",
        attn_dropout=0.0, dropout=0.0, compute_dtype="fp32",
        max_len=64, seed=0, depth=2, d_ff=128,
        linformer_k=16, linformer_sharing="layerwise",
        sparse_pattern="fixed", sparse_stride=8,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_batch(batch=3, seq_len=32, d_model=64, seed=0):
    """Builds a batch with variable sequence lengths to exercise padding."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, d_model, generator=generator)
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, length in enumerate([seq_len, seq_len - 9, 4][:batch]):
        mask[i, :length] = 1
    return x, mask


# Reference arms: the previous implementations, copied unchanged.

class ReferenceVanilla(AttentionBase):
    """The vanilla `_attend` as it stood before the in-place rewrite."""

    def _attend(self, q, k, v, mask):
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            keep = mask[:, None, None, :]
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return weights @ v


class ReferenceSparse(SparseAttention):
    """The sparse `_attend` as it stood before the in-place rewrite.

    It builds the pattern the old way too -- per call, at the padded length -- so the
    comparison holds the pattern fixed and varies only how the mask is applied. The
    tests below pin `sparse_stride` explicitly for that reason.
    """

    def _reference_pattern(self, seq_len, device):
        stride = self.stride or max(1, round(math.sqrt(seq_len)))
        builder = PATTERNS[self.pattern_name]
        mask = torch.stack(
            [builder(seq_len, stride, head) for head in range(self.n_heads)]
        ).to(device)
        diagonal = torch.eye(seq_len, dtype=torch.bool, device=device)
        return mask | diagonal[None]

    def _attend(self, q, k, v, mask):
        seq_len = q.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        allowed = self._reference_pattern(seq_len, q.device)[None]
        if mask is not None:
            allowed = allowed & mask[:, None, None, :]

        empty_rows = ~allowed.any(dim=-1, keepdim=True)
        if empty_rows.any():
            rescue = torch.zeros_like(allowed)
            rescue[..., 0] = True
            allowed = allowed | (empty_rows & rescue)

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        return scores.softmax(dim=-1) @ v


class ReferenceLinformer(LinformerAttention):
    """Linformer scoring as it stood before the in-place rewrite."""

    def _attend(self, q, k, v, mask):
        seq_len = q.shape[2]
        if mask is not None:
            keep = mask[:, None, :, None].to(k.dtype)
            k = k * keep
            v = v * keep
        k_projected = self.key_projection(k, seq_len)
        v_projected = self.value_projection(v, seq_len)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k_projected.transpose(-2, -1)) * scale
        weights = scores.softmax(dim=-1)
        return weights @ v_projected


PAIRS = [
    ("vanilla", VanillaAttention, ReferenceVanilla, {}),
    ("sparse", SparseAttention, ReferenceSparse, {"attention": "sparse"}),
    ("linformer", LinformerAttention, ReferenceLinformer, {"attention": "linformer"}),
]


def build_pair(new_class, reference_class, **cfg_overrides):
    """Constructs an optimised arm and its reference with identical weights."""
    torch.manual_seed(1903697)
    new = new_class(make_cfg(**cfg_overrides)).eval()
    torch.manual_seed(1903697)
    reference = reference_class(make_cfg(**cfg_overrides)).eval()
    reference.load_state_dict(new.state_dict())
    return new, reference


@pytest.mark.parametrize("name,new_class,reference_class,overrides", PAIRS)
def test_forward_matches_previous_implementation(name, new_class, reference_class, overrides):
    """The rewritten arm returns what the previous implementation returned."""
    new, reference = build_pair(new_class, reference_class, **overrides)
    x, mask = make_batch()

    with torch.no_grad():
        difference = (new(x, mask) - reference(x, mask)).abs().max().item()

    print(f"\n{name}: max |new - previous| = {difference:.3e}")
    assert difference < 1e-5, f"{name} changed its output: {difference:.3e}"


@pytest.mark.parametrize("name,new_class,reference_class,overrides", PAIRS)
def test_gradients_match_previous_implementation(name, new_class, reference_class, overrides):
    """The rewritten arm produces the same gradients, so training is unaffected."""
    new, reference = build_pair(new_class, reference_class, **overrides)
    x, mask = make_batch()

    grads = {}
    for label, arm in (("new", new), ("previous", reference)):
        inputs = x.clone().requires_grad_(True)
        arm.zero_grad(set_to_none=True)
        arm(inputs, mask).pow(2).sum().backward()
        grads[label] = (
            inputs.grad,
            {n: p.grad.clone() for n, p in arm.named_parameters()},
        )

    input_difference = (grads["new"][0] - grads["previous"][0]).abs().max().item()
    print(f"\n{name}: max input-grad difference = {input_difference:.3e}")
    assert input_difference < 1e-5, f"{name} input gradient changed: {input_difference:.3e}"

    for parameter_name, grad in grads["new"][1].items():
        difference = (grad - grads["previous"][1][parameter_name]).abs().max().item()
        assert difference < 1e-5, (
            f"{name}.{parameter_name} gradient changed: {difference:.3e}"
        )


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_sparse_matches_previous_for_every_pattern(pattern):
    """Every pattern, not just the configured one, survives the rewrite unchanged."""
    new, reference = build_pair(
        SparseAttention, ReferenceSparse, attention="sparse", sparse_pattern=pattern
    )
    x, mask = make_batch()
    with torch.no_grad():
        difference = (new(x, mask) - reference(x, mask)).abs().max().item()
    assert difference < 1e-5, f"{pattern} changed: {difference:.3e}"


# The in-place softmax itself.

def test_softmax_matches_torch():
    """softmax_ agrees with torch's softmax in value and in gradient."""
    torch.manual_seed(0)
    base = torch.randn(2, 4, 30, 30)

    reference_input = base.clone().requires_grad_(True)
    reference = reference_input.softmax(dim=-1)
    reference.pow(2).sum().backward()

    scores = base.clone().requires_grad_(True)
    result = softmax_(scores * 1.0, dim=-1)     # multiply so the input is not a leaf
    result.pow(2).sum().backward()

    assert (result - reference).abs().max().item() < 1e-6
    assert (scores.grad - reference_input.grad).abs().max().item() < 1e-6


def test_softmax_reuses_the_score_buffer():
    """The point of the helper: no second attention-sized tensor is allocated."""
    scores = torch.randn(2, 3, 8, 8)
    pointer = scores.data_ptr()
    result = softmax_(scores, dim=-1)
    assert result.data_ptr() == pointer, "softmax_ allocated a new buffer"


def test_softmax_handles_a_fully_masked_row():
    """A row of finfo.min normalises to uniform rather than NaN, as before."""
    scores = torch.full((1, 1, 3, 5), torch.finfo(torch.float32).min)
    result = softmax_(scores, dim=-1)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.full_like(result, 0.2))


# Memory behaviour.

class RecordLargeBoolTensors(TorchDispatchMode):
    """Records boolean tensors larger than a per-head pattern, i.e. batch-expanded ones."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in out if isinstance(out, (list, tuple)) else [out]:
            if (isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bool
                    and tensor.numel() > self.threshold):
                self.seen.append(tuple(tensor.shape))
        return out


def test_sparse_never_materialises_a_batched_boolean_mask():
    """The pattern broadcasts over the batch; expanding it would cost more than the scores."""
    batch, seq_len = 3, 32
    arm = SparseAttention(make_cfg(attention="sparse", max_len=seq_len)).eval()
    x, mask = make_batch(batch=batch, seq_len=seq_len)
    heads = arm.n_heads

    # Build the pattern before recording: it is a one-time (H, S, S) allocation, and
    # what this test is about is what each batch costs on top of it.
    arm.pattern_mask(seq_len, x.device)

    # Anything above one per-head pattern but below a batch-expanded one.
    recorder = RecordLargeBoolTensors(threshold=2 * heads * seq_len * seq_len)
    assert batch * heads * seq_len * seq_len > recorder.threshold, "threshold mis-sized"

    with torch.no_grad(), recorder:
        arm(x, mask)

    assert not recorder.seen, (
        f"sparse allocated batch-sized boolean masks: {recorder.seen}; "
        f"the pattern is (H, S, S) and must broadcast over the batch"
    )


def test_sparse_pattern_cache_does_not_grow_with_padded_length():
    """Dynamic padding produces many distinct lengths; only one full pattern may exist."""
    arm = SparseAttention(make_cfg(attention="sparse", max_len=64))
    device = torch.device("cpu")

    for seq_len in range(8, 64):
        arm.pattern_mask(seq_len, device)

    storages = {arm.pattern_mask(n, device).untyped_storage().data_ptr()
                for n in range(8, 64)}
    assert len(storages) == 1, (
        f"{len(storages)} distinct pattern tensors are cached; they should all be "
        "views of one mask built at max_len"
    )


def test_sparse_rejects_a_sequence_longer_than_max_len():
    """The pattern is built at max_len, so a longer batch must fail rather than silently rebuild."""
    arm = SparseAttention(make_cfg(attention="sparse", max_len=32))
    with pytest.raises(ValueError, match="exceeds max_len"):
        arm.pattern_mask(64, torch.device("cpu"))


def test_auto_stride_resolves_from_max_len_not_batch_length():
    """Auto stride must not depend on how the batch happened to pad.

    This is a deliberate change. Resolving sqrt(n) against the padded batch length made
    the attention pattern a function of batch composition, which is a nuisance variable
    in the intervention itself.
    """
    arm = SparseAttention(make_cfg(attention="sparse", max_len=400, sparse_stride=0))
    assert arm.stride == 20

    short = arm.pattern_mask(64, torch.device("cpu"))
    long = arm.pattern_mask(400, torch.device("cpu"))
    torch.testing.assert_close(short, long[:, :64, :64])
