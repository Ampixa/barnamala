import torch

from devnet.models.student import DevNet


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_default_student_under_param_budget():
    model = DevNet()  # widths (40, 80, 160), depths (2, 2, 2)
    assert n_params(model) <= 1_500_000  # spec C2 budget


def test_output_shape():
    model = DevNet()
    x = torch.randn(4, 1, 32, 32)
    assert model(x).shape == (4, 46)


def test_teacher_configuration_scales_up():
    teacher = DevNet(widths=(96, 192, 384), depths=(3, 3, 3))
    assert n_params(teacher) > 5_000_000
    x = torch.randn(2, 1, 32, 32)
    assert teacher(x).shape == (2, 46)


def test_gradients_flow():
    model = DevNet(widths=(8, 16, 32), depths=(1, 1, 1))
    out = model(torch.randn(2, 1, 32, 32))
    out.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
