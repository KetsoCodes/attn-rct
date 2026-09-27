"""Unit tests for the learning-rate warmup.

"""

import torch

from attn_rct.train import apply_warmup


def make_optimizer(lr=1e-3):
    """Creates a dummy optimizer for testing LR scheduling."""
    parameter = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.Adam([parameter], lr=lr)


def test_disabled_warmup_leaves_the_optimizer_untouched():
    """warmup_steps=0 must not write to the optimizer at all."""
    optimizer = make_optimizer(1e-3)
    before = optimizer.param_groups[0]["lr"]
    
    returned = apply_warmup(optimizer, 1e-3, global_step=500, warmup_steps=0)
    
    assert returned == 1e-3
    assert optimizer.param_groups[0]["lr"] == before


def test_ramp_starts_near_zero():
    """The first step gets a very small fraction of the base rate."""
    optimizer = make_optimizer()
    lr = apply_warmup(optimizer, 1e-3, global_step=0, warmup_steps=4000)
    assert 0 < lr < 1e-3 / 100


def test_ramp_is_linear():
    """Halfway through the ramp, the rate is exactly half the target."""
    optimizer = make_optimizer()
    half = apply_warmup(optimizer, 1e-3, global_step=1999, warmup_steps=4000)
    assert abs(half - 5e-4) < 1e-9


def test_ramp_reaches_base_lr_at_the_end():
    """The target learning rate is reached exactly at the final warmup step."""
    optimizer = make_optimizer()
    assert apply_warmup(optimizer, 1e-3, global_step=3999, warmup_steps=4000) == 1e-3


def test_ramp_holds_base_lr_after_the_end():
    """The rate stays constant after the warmup phase ends; no overshooting."""
    optimizer = make_optimizer()
    for step in (4000, 10_000, 1_000_000):
        assert apply_warmup(optimizer, 1e-3, global_step=step, warmup_steps=4000) == 1e-3


def test_ramp_writes_through_to_the_optimizer():
    """The calculated rate is actually applied to the optimizer's param groups."""
    optimizer = make_optimizer()
    lr = apply_warmup(optimizer, 2e-3, global_step=999, warmup_steps=2000)
    assert optimizer.param_groups[0]["lr"] == lr


def test_ramp_is_monotone_non_decreasing():
    """The learning rate never dips during the warmup phase."""
    optimizer = make_optimizer()
    rates = [apply_warmup(optimizer, 1e-3, s, 4000) for s in range(0, 6000, 137)]
    assert all(b >= a for a, b in zip(rates, rates[1:]))


def test_ramp_scales_with_base_lr():
    """Different base rates ramp proportionally."""
    optimizer = make_optimizer()
    low = apply_warmup(optimizer, 1e-4, global_step=999, warmup_steps=4000)
    high = apply_warmup(optimizer, 1e-3, global_step=999, warmup_steps=4000)
    assert abs(high / low - 10.0) < 1e-6