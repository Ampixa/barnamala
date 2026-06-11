# tests/test_train_components.py
import numpy as np
import torch

from devnet.train import EMA, apply_mix, cosine_warmup_scheduler


def test_apply_mix_identity_when_prob_zero():
    x = torch.randn(8, 1, 32, 32)
    y = torch.arange(8)
    rng = np.random.default_rng(0)
    out_x, ya, yb, lam = apply_mix(x.clone(), y, mixup_alpha=0.2,
                                   cutmix_alpha=1.0, mix_prob=0.0, rng=rng)
    assert torch.equal(out_x, x)
    assert torch.equal(ya, y) and torch.equal(yb, y)
    assert lam == 1.0


def test_apply_mix_lambda_in_unit_interval():
    x = torch.randn(8, 1, 32, 32)
    y = torch.arange(8)
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, _, _, lam = apply_mix(x.clone(), y, mixup_alpha=0.2,
                                 cutmix_alpha=1.0, mix_prob=1.0, rng=rng)
        assert 0.0 <= lam <= 1.0


def test_ema_converges_to_constant_weights():
    model = torch.nn.Linear(4, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    for _ in range(60):
        ema.update(model)
    target = torch.nn.Linear(4, 2)
    ema.copy_to(target)
    for p in target.parameters():
        assert torch.allclose(p, torch.ones_like(p), atol=1e-6)


def test_cosine_warmup_shape():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]            # warming up
    assert max(lrs) <= 1.0 + 1e-8     # peak at base lr
    assert lrs[-1] < 0.01             # decayed to ~0
