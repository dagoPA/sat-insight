"""The three head parameterisations agree on shapes, decoding and loss semantics."""

import numpy as np
import pytest

from satinsight import backbone, llp


@pytest.mark.parametrize("head", backbone.HEADS)
def test_every_head_returns_four_cumulative_shares(monkeypatch, head):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(backbone, "HEAD", head)
    torch.manual_seed(0)
    model = llp.build(6, standardize=True)
    shares, per_instance = model(torch.randn(50, 6))
    assert shares.shape == (4,)
    assert per_instance.shape == (50, 4)
    assert torch.all(per_instance >= 0) and torch.all(per_instance <= 1)


@pytest.mark.parametrize("head", ("coral", "softmax"))
def test_ordered_heads_give_ordered_cumulative_shares(monkeypatch, head):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(backbone, "HEAD", head)
    torch.manual_seed(1)
    model = llp.build(6)
    _, per_instance = model(torch.randn(200, 6) * 3)
    assert torch.all(per_instance[:, :-1] >= per_instance[:, 1:] - 1e-6)


def test_independent_thresholds_can_cross(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(backbone, "HEAD", "cumulative")
    torch.manual_seed(2)
    model = llp.build(6)
    with torch.no_grad():
        model.score.bias.copy_(torch.tensor([-2.0, 2.0, -2.0, 2.0]))
        model.score.weight.zero_()
    _, per_instance = model(torch.randn(5, 6))
    assert torch.all(per_instance[:, 0] < per_instance[:, 1])


def test_class_shares_invert_the_cumulative_shares():
    torch = pytest.importorskip("torch")
    cumulative = torch.tensor([[0.9, 0.6, 0.3, 0.1]])
    classes = llp.class_shares(cumulative)
    assert torch.allclose(classes, torch.tensor([[0.1, 0.3, 0.3, 0.2, 0.1]]))
    assert torch.allclose(classes.sum(), torch.tensor(1.0))


def test_softmax_loss_is_cross_entropy_on_class_shares(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(backbone, "HEAD", "softmax")
    target = torch.tensor([0.9, 0.6, 0.3, 0.1])
    assert llp.bag_loss(target.clone(), target).item() == pytest.approx(
        float(-(llp.class_shares(target) * llp.class_shares(target).log()).sum())
    )
    worse = torch.tensor([0.5, 0.4, 0.3, 0.2])
    assert llp.bag_loss(worse, target) > llp.bag_loss(target.clone(), target)
    monkeypatch.setattr(backbone, "HEAD", "cumulative")
    assert llp.bag_loss(target.clone(), target).item() == pytest.approx(
        float(torch.nn.functional.binary_cross_entropy(target, target))
    )


def test_expected_grade_is_the_sum_of_cumulative_shares():
    per_instance = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.9, 0.6, 0.3, 0.1]])
    assert llp.instance_scores(per_instance).tolist() == pytest.approx([4.0, 0.0, 1.9])


def test_suffix_carries_tag_and_head():
    assert backbone.suffix_of("data/predictions_val_ov_coral.parquet") == "_ov_coral"
    assert backbone.suffix_of("data/predictions_test_dofalov.parquet") == "_dofalov"
    assert backbone.suffix_of("data/predictions_val.parquet") == ""
