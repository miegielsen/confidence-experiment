"""Small real encoder update and frozen-head checks, without large downloads."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_modernbert_training_and_saved_model_reload(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from confidence_experiment.models import train_encoder

    torch.set_num_threads(1)
    raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "message": 2, "a": 3, "b": 4, "c": 5}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]")
    configuration = transformers.ModernBertConfig(vocab_size=8, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=64, local_attention=4,
        global_attn_every_n_layers=1, num_labels=3, pad_token_id=1, cls_token_id=2)
    original_load = transformers.AutoModelForSequenceClassification.from_pretrained
    def model_load(source, **kwargs):
        if str(source) == "tiny-model":
            return transformers.ModernBertForSequenceClassification(configuration)
        return original_load(source, **kwargs)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model_load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **k: SimpleNamespace(sha="unit-test-revision"))
    frame = pd.DataFrame({"text": ["message a", "message b", "message c"] * 2, "label": ["a", "b", "c"] * 2})
    frames = {"fit": frame, "development": frame, "test": frame}
    settings = {"seed": 42, "encoder": {"model": "tiny-model", "device": "cpu", "epochs": 1, "batch_size": 2, "max_length": 16}}
    first, metadata = train_encoder(settings, frames, {"a": 0, "b": 1, "c": 2}, ["development"], tmp_path)
    final, _ = train_encoder(settings, frames, {"a": 0, "b": 1, "c": 2}, ["test"], tmp_path, frozen=True)
    assert first["development"].shape == (6, 3)
    assert np.allclose(first["development"], final["test"], atol=1e-6)
    assert metadata["revision"] == "unit-test-revision"
