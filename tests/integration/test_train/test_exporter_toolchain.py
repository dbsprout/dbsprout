"""Real llama.cpp toolchain integration test for the GGUF exporter (S-066b).

Unlike the unit suite (which mocks **only** the ``subprocess.run`` boundary),
this test runs the *genuine* llama.cpp ``convert_hf_to_gguf.py`` script on a
real — but tiny — Hugging Face model directory and asserts the produced bytes
are a valid GGUF (magic ``GGUF`` + supported version), proving the subprocess
seam wired in PR #88 / S-066b actually converts.

It skips cleanly when the real toolchain is unreachable (no
``DBSPROUT_LLAMA_CONVERT`` / ``convert_hf_to_gguf.py`` resolvable) or when the
HF stack (``transformers`` + ``torch`` + ``safetensors``) needed to *build*
the tiny source model is not installed, so a runner without the toolchain
never hides a regression and the unit suite stays green without it.

No fabricated ``llama_cpp`` conversion symbol is used anywhere — the only
real conversion mechanism is the subprocess to the upstream script.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dbsprout.train import exporter as exporter_mod
from dbsprout.train.exporter import _assert_gguf_magic, _quantize_to_gguf

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def _toolchain_available() -> bool:
    """True iff the real ``convert_hf_to_gguf.py`` resolves (env or known)."""
    try:
        exporter_mod._resolve_converter()
    except RuntimeError:
        return False
    return True


def _build_tiny_hf_model(dest: Path) -> Path:
    """Materialize a minimal real LLaMA-architecture HF model directory.

    A genuine (random-weight) ``LlamaForCausalLM`` with a deliberately tiny
    config — large enough for ``convert_hf_to_gguf.py`` to recognize and
    convert, small enough to run in a few seconds. Returns *dest*.
    """
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        LlamaConfig,
        LlamaForCausalLM,
    )

    dest.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    with torch.no_grad():
        model = LlamaForCausalLM(config)
    model.save_pretrained(str(dest), safe_serialization=True)

    # convert_hf_to_gguf.py needs a tokenizer; ship a minimal byte-level
    # GPT-2-style tokenizer.json so the script can build the GGUF vocab.
    tok_vocab = {chr(i): i for i in range(256)}
    tokenizer_json = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": tok_vocab,
            "merges": [],
            "unk_token": None,
        },
    }
    (dest / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    (dest / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}), encoding="utf-8"
    )
    return dest


@pytest.mark.integration
def test_real_convert_hf_to_gguf_produces_valid_gguf(tmp_path: Path) -> None:
    """Genuine HF -> GGUF conversion yields a file with a valid GGUF header.

    Exercises the real subprocess seam end-to-end (no ``subprocess.run`` mock)
    through ``_quantize_to_gguf``'s convert step and asserts the output is a
    real GGUF via the always-on ``_assert_gguf_magic`` gate.
    """
    if not _toolchain_available():
        pytest.skip(
            "requires a real llama.cpp toolchain "
            "(set DBSPROUT_LLAMA_CONVERT or install convert_hf_to_gguf.py)"
        )
    pytest.importorskip("torch", reason="HF stack needed to build the tiny source model")
    pytest.importorskip("transformers", reason="HF stack needed to build the tiny source model")
    pytest.importorskip("safetensors", reason="safetensors needed for the source model")

    merged = _build_tiny_hf_model(tmp_path / "merged")
    out = tmp_path / "models" / "custom" / "tiny-Q4_K_M.gguf"

    quantizer_ok = True
    try:
        exporter_mod._resolve_quantizer()
    except RuntimeError:
        quantizer_ok = False

    if quantizer_ok:
        # Full real path: convert (f16) + real llama-quantize -> Q4_K_M.
        result = _quantize_to_gguf(merged, out)
        assert result == out.resolve()
        _assert_gguf_magic(result)  # genuine GGUF bytes on disk
        assert result.stat().st_size > 0
    else:
        # No quantizer binary: still prove the real converter produced a
        # valid GGUF by invoking the genuine script directly via the same
        # hardened seam (list argv, no shell), then header-validating it.
        import sys  # noqa: PLC0415

        converter = exporter_mod._resolve_converter()
        f16 = (tmp_path / "tiny-f16.gguf").resolve()
        exporter_mod._run_tool(
            [
                sys.executable,
                str(converter),
                str(merged.resolve()),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ]
        )
        _assert_gguf_magic(f16)  # genuine GGUF bytes from the real script
        assert f16.stat().st_size > 0
