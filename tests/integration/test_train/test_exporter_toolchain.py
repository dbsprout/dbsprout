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


_TOKENIZER_VOCAB_SIZE = 256


def _write_tiny_spm_tokenizer(dest: Path) -> None:
    """Train a genuine minimal SentencePiece ``tokenizer.model`` into *dest*.

    The pinned llama.cpp ``convert_hf_to_gguf.py`` (tag ``b9174``) routes a
    LLaMA model through ``LlamaModel.set_vocab`` -> ``_set_vocab_sentencepiece``
    *first*, which only needs a real SPM ``tokenizer.model`` and reads its
    tokens/scores/types directly. This deliberately avoids the two failure
    modes of a synthetic ``tokenizer.json``:

    * ``transformers``'s ``PreTrainedTokenizerFast`` unconditionally does
      ``tokenizer_file_handle.pop("added_tokens")`` when loading a fast
      tokenizer (``KeyError: 'added_tokens'`` on a hand-rolled file), and
    * the GPT-2/BPE fallback hashes the tokenizer and raises
      ``NotImplementedError("BPE pre-tokenizer was not recognized")`` for any
      tokenizer not in the converter's hard-coded known-hash list.

    A real (tiny) SPM model sidesteps both: no fast-tokenizer load, no hash
    recognition. ``sentencepiece`` is a pinned converter dependency, so it is
    importable wherever this integration test actually runs.
    """
    import sentencepiece as spm  # noqa: PLC0415

    corpus = dest / "_spm_corpus.txt"
    corpus.write_text(
        (
            "the quick brown fox jumps over the lazy dog\n"
            "hello world dbsprout tiny model fixture\n"
            "abcdefghijklmnopqrstuvwxyz 0123456789\n"
        )
        * 50,
        encoding="utf-8",
    )
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(dest / "_spm"),
        vocab_size=_TOKENIZER_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
    )
    (dest / "_spm.model").rename(dest / "tokenizer.model")
    (dest / "_spm.vocab").unlink()
    corpus.unlink()


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
        vocab_size=_TOKENIZER_VOCAB_SIZE,
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

    # convert_hf_to_gguf.py needs a tokenizer; ship a genuine minimal
    # SentencePiece tokenizer.model so the converter takes the
    # _set_vocab_sentencepiece path (no transformers fast-tokenizer load
    # that would KeyError on a missing "added_tokens", and no BPE
    # pre-tokenizer hash recognition that rejects synthetic tokenizers).
    _write_tiny_spm_tokenizer(dest)
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
    pytest.importorskip("sentencepiece", reason="sentencepiece needed for the tiny tokenizer")

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
