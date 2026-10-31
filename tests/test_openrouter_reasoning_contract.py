import asyncio

import httpx
import pytest

from provider_adapters import verify_openrouter_catalog
from provider_models import (
    OPENROUTER_MODEL_SPECS,
    build_reasoning_callback_token,
    effective_chat_output_tokens,
    openrouter_endpoint_supports_contract,
    openrouter_request_contracts,
    openrouter_required_parameters,
    resolve_reasoning_callback_token,
    validate_chat_output_tokens,
)


def test_reasoning_routing_contract_is_model_specific():
    auto = openrouter_required_parameters(
        "qwen/qwen3.8-27b",
        "auto",
        temperature_present=True,
        max_output_tokens_present=True,
    )
    maximum = openrouter_required_parameters(
        "qwen/qwen3.8-27b",
        "max",
        temperature_present=True,
        max_output_tokens_present=True,
    )
    endpoint_a = {
        "status": 0,
        "supported_parameters": ["max_tokens", "temperature", "reasoning", "reasoning_effort"],
    }
    endpoint_b = {
        "status": 0,
        "supported_parameters": ["max_tokens", "temperature"],
    }

    assert auto.requires_parameter_routing is False
    assert maximum.requires_parameter_routing is True
    assert openrouter_endpoint_supports_contract(endpoint_a, maximum)
    assert not openrouter_endpoint_supports_contract(endpoint_b, maximum)
    assert openrouter_endpoint_supports_contract(endpoint_b, auto)


def test_qwen_request_ids_are_not_replaced_by_live_snapshot_slugs():
    assert OPENROUTER_MODEL_SPECS["qwen/qwen3.7-plus"].model_id == "qwen/qwen3.7-plus"
    assert OPENROUTER_MODEL_SPECS["qwen/qwen3.8-27b"].model_id == "qwen/qwen3.8-27b"
    qwen37 = build_reasoning_callback_token("OpenRouter", "qwen/qwen3.7-plus", "enabled")
    qwen38 = build_reasoning_callback_token("OpenRouter", "qwen/qwen3.8-27b", "max")
    assert qwen37 != qwen38
    assert resolve_reasoning_callback_token(qwen37) == ("OpenRouter", "qwen/qwen3.7-plus", "enabled")
    assert resolve_reasoning_callback_token(qwen38) == ("OpenRouter", "qwen/qwen3.8-27b", "max")


def test_qwen_output_floor_applies_to_auto_and_explicit_modes():
    assert effective_chat_output_tokens("OpenRouter", "qwen/qwen3.8-27b", None) == 32768
    assert effective_chat_output_tokens("OpenRouter", "qwen/qwen3.8-27b", 32768) == 32768
    assert validate_chat_output_tokens("OpenRouter", "qwen/qwen3.8-27b", 32768) == 32768
    with pytest.raises(ValueError):
        validate_chat_output_tokens("OpenRouter", "qwen/qwen3.8-27b", 32769)


def _qwen38_catalog_item():
    return {
        "id": "qwen/qwen3.8-27b",
        "canonical_slug": "qwen/qwen3.8-27b-20260814",
        "architecture": {"input_modalities": ["text", "image", "video"]},
        "context_length": 1000000,
        "top_provider": {"max_completion_tokens": 235929},
        "supported_parameters": ["max_tokens", "temperature", "reasoning", "reasoning_effort"],
        "reasoning": {
            "mandatory": False,
            "default_enabled": True,
            "default_effort": "xhigh",
            "supported_efforts": ["xhigh", "medium", "low", "none"],
        },
    }


def test_catalog_verifier_uses_endpoint_safe_floor_not_top_provider_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [_qwen38_catalog_item()]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [
                            {
                                "status": 0,
                                "max_completion_tokens": 32768,
                                "supported_parameters": [
                                    "max_tokens",
                                    "temperature",
                                    "reasoning",
                                    "reasoning_effort",
                                ],
                            },
                            {
                                "status": 0,
                                "max_completion_tokens": 235929,
                                "supported_parameters": ["max_tokens", "temperature"],
                            },
                        ]
                    }
                },
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    result = asyncio.run(verify_openrouter_catalog(transport=httpx.MockTransport(handler)))
    evidence = result["qwen/qwen3.8-27b"]

    assert evidence["request_model_id"] == "qwen/qwen3.8-27b"
    assert evidence["canonical_slug"] == "qwen/qwen3.8-27b-20260814"
    assert evidence["top_provider_output_limit"] == 235929
    assert evidence["observed_top_provider_output_limit"] == 235929
    assert evidence["output_limit"] == 32768
    assert evidence["endpoint_safe_output_limit"] == 32768
    assert evidence["configured_output_limit"] == 32768
    assert evidence["endpoint_safe_output_limits_by_mode"]["auto"] == 32768
    assert evidence["endpoint_safe_output_limits_by_mode"]["max"] == 32768
    assert evidence["compatible_endpoint_counts_by_mode"]["auto"] == 2
    assert evidence["compatible_endpoint_counts_by_mode"]["max"] == 1
    assert evidence["supported_efforts_present"] is True
    assert evidence["reasoning_supports_max_tokens_present"] is False
    assert evidence["reasoning_supports_max_tokens"] is None
    assert evidence["matches_static"] is True
    assert "configured_output_limit" not in evidence["mismatches"]


def test_explicit_reasoning_endpoint_matrix_excludes_endpoint_without_reasoning():
    contracts = openrouter_request_contracts("qwen/qwen3.8-27b")
    explicit = [contract for contract in contracts if contract.reasoning_mode == "max"]
    assert explicit
    assert all(contract.requires_parameter_routing for contract in explicit)
    assert all("reasoning" in contract.required_parameters for contract in explicit)


def test_qwen37_endpoint_safe_floor_is_verified_separately_from_top_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [{
                        "id": "qwen/qwen3.7-plus",
                        "canonical_slug": "qwen/qwen3.7-plus-20260602",
                        "architecture": {"input_modalities": ["text", "image"]},
                        "context_length": 1000000,
                        "top_provider": {"max_completion_tokens": 131072},
                        "supported_parameters": ["max_tokens", "temperature", "reasoning"],
                        "reasoning": {"mandatory": False, "default_enabled": True},
                    }],
                },
            )
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [
                            {
                                "status": 0,
                                "max_completion_tokens": 131072,
                                "supported_parameters": ["max_tokens", "temperature", "reasoning"],
                            },
                            {
                                "status": 0,
                                "max_completion_tokens": 262144,
                                "supported_parameters": ["max_tokens", "temperature", "reasoning"],
                            },
                        ]
                    }
                },
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    result = asyncio.run(verify_openrouter_catalog(transport=httpx.MockTransport(handler)))
    evidence = result["qwen/qwen3.7-plus"]
    assert evidence["request_model_id"] == "qwen/qwen3.7-plus"
    assert evidence["canonical_slug"] == "qwen/qwen3.7-plus-20260602"
    assert evidence["top_provider_output_limit"] == 131072
    assert evidence["endpoint_safe_output_limit"] == 131072
    assert evidence["configured_output_limit"] == 131072
    assert evidence["supported_efforts"] is None
    assert evidence["supported_efforts_present"] is False
    assert evidence["reasoning_supports_max_tokens_present"] is False
    assert evidence["reasoning_supports_max_tokens"] is None
    assert evidence["mandatory"] is False
    assert evidence["default_enabled"] is True
    assert evidence["can_disable"] is True
    assert evidence["matches_static"] is True


def test_catalog_verifier_distinguishes_null_efforts_from_omitted_efforts():
    item = _qwen38_catalog_item()
    item["reasoning"]["supported_efforts"] = None

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [item]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [{
                            "status": 0,
                            "max_completion_tokens": 32768,
                            "supported_parameters": [
                                "max_tokens",
                                "temperature",
                                "reasoning",
                                "reasoning_effort",
                            ],
                        }]
                    }
                },
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    result = asyncio.run(verify_openrouter_catalog(transport=httpx.MockTransport(handler)))
    evidence = result["qwen/qwen3.8-27b"]
    assert evidence["supported_efforts_present"] is True
    assert evidence["supported_efforts"] is None
    assert "reasoning/supported_efforts" in evidence["mismatches"]


def test_catalog_verifier_fails_closed_when_compatible_endpoint_limit_is_unknown():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [_qwen38_catalog_item()]})
        if request.url.path.endswith("/endpoints"):
            params = ["max_tokens", "temperature", "reasoning", "reasoning_effort"]
            return httpx.Response(
                200,
                json={"data": {"endpoints": [
                    {"status": 0, "max_completion_tokens": 32768, "supported_parameters": params},
                    {"status": 0, "supported_parameters": params},
                ]}},
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    result = asyncio.run(verify_openrouter_catalog(transport=httpx.MockTransport(handler)))
    evidence = result["qwen/qwen3.8-27b"]
    assert evidence["endpoint_safe_output_limit"] is None
    assert evidence["endpoint_safe_output_limits_by_mode"]["auto"] is None
    assert evidence["endpoint_safe_output_limits_by_mode"]["max"] is None
    assert "endpoint_safe_output_limit" in evidence["mismatches"]


def test_kimi_portable_limit_matches_active_endpoint_floor():
    item = {
        "id": "moonshotai/kimi-k2.6",
        "canonical_slug": "moonshotai/kimi-k2.6-20260420",
        "architecture": {"input_modalities": ["text", "image"]},
        "context_length": 262144,
        "top_provider": {"max_completion_tokens": 235929},
        "supported_parameters": ["max_tokens", "temperature", "tools"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [item]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200,
                json={"data": {"endpoints": [
                    {"status": 0, "max_completion_tokens": 16384, "supported_parameters": ["max_tokens", "temperature"]},
                    {"status": 0, "max_completion_tokens": 235929, "supported_parameters": ["max_tokens", "temperature"]},
                ]}},
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    result = asyncio.run(verify_openrouter_catalog(transport=httpx.MockTransport(handler)))
    evidence = result["moonshotai/kimi-k2.6"]
    assert evidence["configured_output_limit"] == 16384
    assert evidence["endpoint_safe_output_limit"] == 16384
    assert evidence["top_provider_output_limit"] == 235929
    assert evidence["matches_static"] is True
