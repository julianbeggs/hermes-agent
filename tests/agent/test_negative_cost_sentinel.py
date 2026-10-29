"""A negative cost is a provider sentinel, never a price.

openrouter/auto (and other variable-cost aliases) publish ``-1`` for routes the
provider cannot quote per token. That sentinel used to flow straight into
``estimated_cost_usd``; ``hermes_state_runtime._worker_usage`` rejects values
outside ``[0, 1e12]``, so a managed worker crashed with
``WorkerPersistenceError: invalid_params`` on its very first usage write.

Three layers are guarded: pricing reports ``None``/``unknown``, the turn-usage
delta never queues a negative, and the state runtime sanitizes old poisoned
payloads instead of refusing them.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _no_conversation_loop_boot(monkeypatch):
    """``_loop_mod()`` imports ``agent.conversation_loop``, which boots the CLI
    dependency selector; these tests only need the compression-budget predicate."""
    monkeypatch.setattr(
        "agent.turn_usage._loop_mod",
        lambda: SimpleNamespace(_should_rearm_compression_budget=lambda *_a, **_k: False),
    )


# ---------------------------------------------------------------------------
# pricing layer
# ---------------------------------------------------------------------------

def test_provider_reported_negative_amount_is_unknown_cost(monkeypatch):
    """A profile that hands back a negative quote (the OpenRouter sentinel) is unknown."""
    import providers
    from agent.usage_pricing import CanonicalUsage, CostResult, estimate_usage_cost

    class _Profile:
        def get_usage_cost(self, model_name, usage):
            return CostResult(
                amount_usd=Decimal(-1), status="estimated",
                source="provider_models_api", label="$0.00",
            )

    monkeypatch.setattr(providers, "get_provider_profile", lambda _provider: _Profile())

    result = estimate_usage_cost(
        "openrouter/auto", CanonicalUsage(input_tokens=1_000, output_tokens=100), provider="openrouter",
    )

    assert result.amount_usd is None
    assert result.status == "unknown"
    assert any("sentinel" in note for note in result.notes)


def test_negative_per_token_rate_is_a_sentinel_not_a_price(monkeypatch):
    """A ``-1`` rate card (not a negative provider quote) is also reported unknown."""
    import providers
    from agent import usage_pricing as up

    monkeypatch.setattr(providers, "get_provider_profile", lambda _provider: None)
    monkeypatch.setattr(
        up, "resolve_billing_route",
        lambda *_a, **_k: up.BillingRoute(provider="openrouter", model="auto",
                                          base_url="https://openrouter.ai/api/v1"),
    )
    monkeypatch.setattr(
        up, "get_pricing_entry",
        lambda *_a, **_k: up.PricingEntry(
            input_cost_per_million=Decimal(-1), output_cost_per_million=Decimal(-1),
            source="provider_models_api",
        ),
    )

    result = up.estimate_usage_cost(
        "openrouter/auto", up.CanonicalUsage(input_tokens=1_000, output_tokens=1_000), provider="openrouter",
    )

    assert result.amount_usd is None
    assert result.status == "unknown"
    assert any("sentinel" in note for note in result.notes)


# ---------------------------------------------------------------------------
# turn-usage layer
# ---------------------------------------------------------------------------

class _Compressor:
    threshold_tokens = 0
    _verify_compaction_cleared_threshold = False
    _context_probed = False

    def update_from_response(self, usage):
        pass


class _SessionDB:
    """Records what the turn queues instead of writing a real state.db."""

    def __init__(self):
        self.queued = []

    def queue_token_counts(self, session_id, **kwargs):
        self.queued.append(dict(kwargs, session_id=session_id))


def _agent():
    """A stub agent: the real AIAgent boot reads the developer's ~/.hermes, which a
    unit test must not touch."""
    db = _SessionDB()
    return SimpleNamespace(
        context_compressor=_Compressor(),
        session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0, session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0, session_cost_status=None, session_cost_source=None,
        model="openrouter/auto", provider="openrouter", api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1", api_key="k", platform="cli",
        client=None, _session_db=db, session_id="t", _session_db_created=True,
        verbose_logging=False, quiet_mode=True, log_prefix="", _vprint=lambda *a, **k: None,
        _api_latency_history=None, _api_output_history=None,
    )


def _response(prompt=1_000, completion=10):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion,
                              prompt_tokens_details=None, completion_tokens_details=None),
        id="gen-negative-cost", model="openrouter/auto",
    )


def _record(agent):
    from agent import turn_usage
    return turn_usage.record_response_usage(
        agent, _response(), messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=0.1, compression_attempts=0, max_compression_attempts=3,
    )


def test_negative_delta_never_reaches_queue_token_counts(monkeypatch):
    """Even if pricing somehow yields a negative amount, the queued delta is None."""
    agent = _agent()
    monkeypatch.setattr(
        "agent.turn_usage.estimate_usage_cost",
        lambda *_a, **_k: SimpleNamespace(
            amount_usd=Decimal(-1), status="estimated",
            source="provider_models_api", label="$0.00", notes=(),
        ),
    )

    _record(agent)  # must not raise

    queued = agent._session_db.queued[-1]
    assert queued["estimated_cost_usd"] is None
    assert queued["cost_status"] == "unknown"
    # The running session total is not poisoned by the sentinel.
    assert agent.session_estimated_cost_usd == 0.0


def test_openrouter_sentinel_end_to_end_records_unknown_cost(monkeypatch):
    """The real pricing path with a negative provider quote persists no negative cost."""
    import providers
    from agent.usage_pricing import CostResult

    agent = _agent()
    monkeypatch.setattr(
        providers, "get_provider_profile",
        lambda _provider: SimpleNamespace(get_usage_cost=lambda *_a: CostResult(
            amount_usd=Decimal(-1), status="estimated",
            source="provider_models_api", label="$0.00",
        )),
    )

    _record(agent)

    queued = agent._session_db.queued[-1]
    assert queued["estimated_cost_usd"] is None
    assert queued["cost_status"] == "unknown"
    assert agent.session_estimated_cost_usd == 0.0


def test_positive_cost_still_queues(monkeypatch):
    """The guard is not a blanket drop: a real quote still sets the total and delta."""
    from agent.usage_pricing import CostResult

    agent = _agent()
    monkeypatch.setattr(
        "agent.turn_usage.estimate_usage_cost",
        lambda *_a, **_k: CostResult(
            amount_usd=Decimal("0.25"), status="estimated",
            source="provider_models_api", label="$0.25",
        ),
    )

    _record(agent)

    queued = agent._session_db.queued[-1]
    assert queued["estimated_cost_usd"] == 0.25
    assert queued["cost_status"] == "estimated"
    assert agent.session_estimated_cost_usd == 0.25
