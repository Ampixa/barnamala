# tests/test_config.py
from devnet.config import RunConfig


def test_defaults_match_spec():
    cfg = RunConfig()
    assert cfg.widths == (40, 80, 160)
    assert cfg.label_smoothing == 0.1
    assert cfg.val_fraction == 0.1
    assert cfg.epochs == 300


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 5\nwidths: [8, 16, 32]\nseed: 3\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.epochs == 5
    assert cfg.widths == (8, 16, 32)
    assert cfg.seed == 3
    assert cfg.lr == RunConfig().lr  # unspecified keys keep defaults


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("eposh: 5\n")  # typo must not pass silently
    import pytest
    with pytest.raises(TypeError):
        RunConfig.from_yaml(p)
