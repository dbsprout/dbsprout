"""FAB-API guard: assert the REAL litellm / llama-cpp usage symbols exist.

S-080a wires real token/cost accounting against external optional deps
(``litellm`` for cloud, ``llama-cpp-python`` for embedded). These are the
exact symbols agents tend to fabricate, so this integration test pins the
contract: each test is skipped unless the real dependency is importable,
and asserts the exact attributes the production seam relies on.

Run with the cloud/llm extras installed:
    uv run pytest -m integration tests/integration/test_llm_usage_contract.py
"""

from __future__ import annotations

import importlib.util
from typing import ClassVar

import pytest

pytestmark = pytest.mark.integration

_HAS_LITELLM = importlib.util.find_spec("litellm") is not None
_HAS_INSTRUCTOR = importlib.util.find_spec("instructor") is not None
_HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None


@pytest.mark.skipif(not _HAS_LITELLM, reason="litellm not installed (cloud extra)")
def test_litellm_completion_cost_symbol_exists() -> None:
    """``litellm.completion_cost`` must exist and accept ``completion_response``."""
    import inspect  # noqa: PLC0415

    import litellm  # noqa: PLC0415

    assert hasattr(litellm, "completion_cost")
    assert callable(litellm.completion_cost)
    sig = inspect.signature(litellm.completion_cost)
    assert "completion_response" in sig.parameters


@pytest.mark.skipif(not _HAS_LITELLM, reason="litellm not installed (cloud extra)")
def test_litellm_usage_object_has_token_attrs() -> None:
    """litellm's Usage object exposes the exact token attributes we read."""
    from litellm.types.utils import Usage  # noqa: PLC0415

    usage = Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 8


@pytest.mark.skipif(
    not (_HAS_LITELLM and _HAS_INSTRUCTOR),
    reason="litellm+instructor not installed (cloud extra)",
)
def test_instructor_from_litellm_and_create_with_completion_exist() -> None:
    """instructor exposes ``from_litellm`` and ``create_with_completion``."""
    import instructor  # noqa: PLC0415
    import litellm  # noqa: PLC0415

    assert hasattr(instructor, "from_litellm")
    client = instructor.from_litellm(litellm.completion)
    assert hasattr(client.chat.completions, "create_with_completion")
    assert callable(client.chat.completions.create_with_completion)


@pytest.mark.skipif(not _HAS_LLAMA_CPP, reason="llama-cpp-python not installed (llm extra)")
def test_llama_cpp_completion_usage_typeddict_keys() -> None:
    """llama-cpp's CompletionUsage carries prompt/completion token keys."""
    from llama_cpp.llama_types import CompletionUsage  # noqa: PLC0415

    keys = set(CompletionUsage.__annotations__)
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= keys


@pytest.mark.skipif(not _HAS_LITELLM, reason="litellm not installed (cloud extra)")
def test_seam_maps_real_usage_object() -> None:
    """The cloud seam maps a real litellm Usage object correctly."""
    from litellm.types.utils import Usage  # noqa: PLC0415

    from dbsprout.spec.providers.cloud import _usage_from_completion  # noqa: PLC0415

    class _RawNoCost:
        usage = Usage(prompt_tokens=12, completion_tokens=34, total_tokens=46)
        _hidden_params: ClassVar[dict[str, object]] = {}

    import litellm  # noqa: PLC0415

    spec_usage = _usage_from_completion(litellm, _RawNoCost())
    assert spec_usage.tokens_sent == 12
    assert spec_usage.tokens_received == 34
    # No cost source on the stub → honest 0.0, never fabricated.
    assert spec_usage.cost_usd == 0.0
