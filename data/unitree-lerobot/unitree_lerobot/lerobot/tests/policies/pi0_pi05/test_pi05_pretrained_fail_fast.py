# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.modeling_pi05 import (
    _load_safetensors_with_aliases,
    _prefer_trained_paligemma_embeddings,
)


@pytest.mark.parametrize("prefix", ["", "model."])
def test_tied_checkpoint_metadata_materializes_embedding_key(tmp_path, prefix):
    lm_head_key = f"{prefix}paligemma_with_expert.paligemma.lm_head.weight"
    embedding_key = f"{prefix}paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    checkpoint = tmp_path / "model.safetensors"
    lm_head = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file(
        {lm_head_key: lm_head},
        checkpoint,
        metadata={embedding_key: lm_head_key, "format": "pt"},
    )

    loaded = _load_safetensors_with_aliases(checkpoint)

    assert torch.equal(loaded[embedding_key], lm_head)
    assert loaded[embedding_key].data_ptr() == loaded[lm_head_key].data_ptr()


def test_tied_checkpoint_alias_target_must_exist(tmp_path):
    embedding_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {"other.weight": torch.ones(1)},
        checkpoint,
        metadata={embedding_key: lm_head_key},
    )

    with pytest.raises(RuntimeError, match="alias target"):
        _load_safetensors_with_aliases(checkpoint)


def test_lm_head_without_alias_metadata_does_not_hide_missing_embedding(tmp_path):
    embedding_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
    checkpoint = tmp_path / "model.safetensors"
    save_file({lm_head_key: torch.ones(2, 3)}, checkpoint)

    loaded = _load_safetensors_with_aliases(checkpoint)

    assert embedding_key not in loaded


def test_untied_checkpoint_prefers_trained_embedding_value():
    embedding_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
    embedding = torch.full((2, 3), 2.0)
    state_dict = {embedding_key: embedding, lm_head_key: torch.full((2, 3), -1.0)}

    _prefer_trained_paligemma_embeddings(state_dict)

    assert state_dict[lm_head_key] is embedding


def test_missing_pretrained_weights_never_returns_random_model():
    config = PI05Config(device="cpu")
    with (
        patch.object(PI05Policy, "__init__", return_value=None),
        patch("transformers.utils.cached_file", side_effect=FileNotFoundError("missing")),
        pytest.raises(RuntimeError, match="Refusing to continue"),
    ):
        PI05Policy.from_pretrained("/missing/pi05", config=config, local_files_only=True)
