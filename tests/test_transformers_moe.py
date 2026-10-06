import pytest

from mindpage.integrations.transformers_moe import TransformersMoETracer


def test_extract_choices_with_fake_tensor_shape_contract() -> None:
    torch = pytest.importorskip("torch")
    logits = (
        torch.tensor([[[0.1, 0.8, 0.2], [0.9, 0.3, 0.1]]]),
        torch.tensor([[[0.4, 0.2, 0.7], [0.1, 0.6, 0.3]]]),
    )
    choices = TransformersMoETracer._extract_choices(logits, 2)
    assert len(choices) == 8
    assert (choices[0].layer, choices[0].token_index, choices[0].expert_id) == (0, 0, 1)
    assert choices[1].expert_id == 2
    assert choices[-2].expert_id == 1
