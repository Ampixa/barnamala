"""Image corruptions for robustness curves (applied to [0,1] floats,
BEFORE normalization). Severity 0 = identity, 1..5 increasing."""
import torch
from torchvision.transforms.v2 import functional as TF

_NOISE_STD = [0.0, 0.04, 0.08, 0.12, 0.18, 0.25]
_BLUR_SIGMA = [0.0, 0.4, 0.7, 1.0, 1.4, 1.9]
_CONTRAST = [1.0, 0.8, 0.65, 0.5, 0.35, 0.2]

CORRUPTIONS = ("noise", "blur", "contrast")


def corrupt(x: torch.Tensor, kind: str, severity: int,
            generator: torch.Generator | None = None) -> torch.Tensor:
    if severity == 0:
        return x.clone()
    if kind == "noise":
        noise = torch.randn(x.shape, generator=generator, device=x.device) * _NOISE_STD[severity]
        return (x + noise).clamp(0.0, 1.0)
    if kind == "blur":
        return TF.gaussian_blur(x, kernel_size=5,
                                sigma=_BLUR_SIGMA[severity]).clamp(0.0, 1.0)
    if kind == "contrast":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (mean + (x - mean) * _CONTRAST[severity]).clamp(0.0, 1.0)
    raise ValueError(f"unknown corruption {kind!r}")
