from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

from manual_portfolio.service import (
    daily_run,
    onboard_portfolio,
    portfolio_status,
    record_fill,
)
from manual_portfolio.storage import (
    append_jsonl,
    load_state,
    read_json,
    read_jsonl,
    tail_jsonl,
    write_json,
)


def _write_registry(
    path: Path,
    *,
    active_strategy_id: str,
    signal_path: Path,
    price_path: Path,
    every_n_signals: int = 1,
    anchor_date: str = "2026-01-01",
    top_n: int = 2,
    gross_exposure: float = 1.0,
    quantity_policy: str = "fractional",
    min_trade_value: float = 0.0,
    min_delta_quantity: float = 0.0,
) -> Path:
    payload = {
        "schema_version": 1,
        "active_strategy_id": active_strategy_id,
        "source_prefix": "test",
        "rebalance_cadence": {
            "kind": "every_n_signal_dates",
            "every_n_signals": every_n_signals,
            "anchor_date": anchor_date,
        },
        "sizing_rule": {
            "kind": "top_n_equal_weight",
            "top_n": top_n,
            "gross_exposure": gross_exposure,
        },
        "execution_policy": {
            "quantity_policy": quantity_policy,
            "min_trade_value": min_trade_value,
            "min_delta_quantity": min_delta_quantity,
        },
        "artifacts": {
            "signal": {
                "path": signal_path.name,
                "date_col": "timestamp",
                "asset_col": "asset",
                "value_col": "signal",
            },
            "price": {
                "path": price_path.name,
                "date_col": "date",
                "asset_col": "asset",
                "value_col": "price",
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _write_market_data(tmp_path: Path) -> tuple[Path, Path]:
    signal_path = tmp_path / "signals.parquet"
    price_path = tmp_path / "prices.parquet"
    pl.DataFrame(
        {
            "timestamp": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 3),
                date(2026, 1, 3),
            ],
            "asset": [
                "AAPL",
                "MSFT",
                "NVDA",
                "AAPL",
                "MSFT",
                "NVDA",
                "AAPL",
                "MSFT",
                "NVDA",
            ],
            "signal": [10.0, 9.0, 1.0, 8.0, 7.0, 2.0, 1.0, 11.0, 10.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime)).write_parquet(signal_path)
    pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 3),
                date(2026, 1, 3),
            ],
            "asset": [
                "AAPL",
                "MSFT",
                "NVDA",
                "AAPL",
                "MSFT",
                "NVDA",
                "AAPL",
                "MSFT",
                "NVDA",
            ],
            "price": [100.0, 200.0, 50.0, 100.0, 200.0, 50.0, 100.0, 200.0, 50.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime)).write_parquet(price_path)
    return signal_path, price_path


def test_onboarding_both_modes(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    cash_only = onboard_portfolio(state_root, "cash", starting_cash=1000.0)
    imported = onboard_portfolio(
        state_root,
        "imported",
        starting_cash=250.0,
        imported_holdings=[{"symbol": "AAPL", "quantity": 2.0, "avg_cost": 90.0}],
    )

    assert cash_only["onboarding_mode"] == "cash_only"
    assert imported["onboarding_mode"] == "import_holdings"
    imported_state = load_state(state_root, "imported")
    assert imported_state.cash == 250.0
    assert imported_state.holdings["AAPL"].quantity == 2.0
    assert imported_state.holdings["AAPL"].avg_cost == 90.0


def test_buy_partial_sell_full_close_updates_ledger_and_realized_pnl(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    onboard_portfolio(state_root, "p1", starting_cash=10_000.0)

    record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 1),
        symbol="AAPL",
        side="buy",
        quantity=10.0,
        fill_price=100.0,
        commission=1.0,
        slippage=1.0,
    )
    record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 2),
        symbol="AAPL",
        side="sell",
        quantity=4.0,
        fill_price=110.0,
        commission=1.0,
        slippage=1.0,
    )
    result = record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 3),
        symbol="AAPL",
        side="sell",
        quantity=6.0,
        fill_price=120.0,
        commission=1.0,
        slippage=1.0,
    )

    state = load_state(state_root, "p1")
    assert state.holdings == {}
    assert round(state.cash, 6) == 10154.0
    assert round(state.realized_pnl, 6) == 154.0
    assert result["holdings"] == {}


def test_add_to_position_blends_cost_basis_with_trade_costs(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    onboard_portfolio(state_root, "p1", starting_cash=10_000.0)

    record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 1),
        symbol="AAPL",
        side="buy",
        quantity=10.0,
        fill_price=100.0,
        commission=1.0,
        slippage=1.0,
    )
    record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 2),
        symbol="AAPL",
        side="buy",
        quantity=5.0,
        fill_price=120.0,
        commission=2.0,
        slippage=3.0,
    )

    state = load_state(state_root, "p1")
    assert round(state.cash, 6) == 8393.0
    assert state.holdings["AAPL"].quantity == 15.0
    assert round(state.holdings["AAPL"].avg_cost, 6) == round(
        (1002.0 + 605.0) / 15.0, 6
    )
    assert state.realized_pnl == 0.0


def test_drift_generation_when_actual_differs_from_target(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)
    record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 1),
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        fill_price=100.0,
    )

    status = portfolio_status(state_root, registry_path, "p1", as_of=date(2026, 1, 1))
    actions = {row["asset"]: row["action"] for row in status["rebalance_plan"]["rows"]}

    assert actions["AAPL"] == "buy"
    assert actions["MSFT"] == "buy"


def test_non_rebalance_day_behavior(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
        every_n_signals=2,
        anchor_date="2026-01-01",
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 2))
    portfolio = result["portfolios"]["p1"]

    assert portfolio["is_rebalance_day"] is False
    assert portfolio["rebalance_plan"]["instructions"] == []
    assert portfolio["rebalance_plan"]["suppressed"] is True


def test_rebalance_day_action_generation(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
        every_n_signals=2,
        anchor_date="2026-01-01",
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 3))
    portfolio = result["portfolios"]["p1"]

    assert portfolio["is_rebalance_day"] is True
    assert portfolio["rebalance_plan"]["instructions"]
    output_dir = state_root / "p1" / "daily" / "2026-01-03"
    assert (output_dir / "daily_summary.md").exists()
    assert (output_dir / "rebalance_plan.json").exists()


def test_multiple_portfolios_share_one_active_strategy(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    onboard_portfolio(state_root, "small", starting_cash=1000.0)
    onboard_portfolio(state_root, "large", starting_cash=2000.0)

    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 1))
    small_positions = {
        row["asset"]: row
        for row in result["portfolios"]["small"]["target_snapshot"]["positions"]
    }
    large_positions = {
        row["asset"]: row
        for row in result["portfolios"]["large"]["target_snapshot"]["positions"]
    }

    assert (
        large_positions["AAPL"]["target_quantity"]
        == small_positions["AAPL"]["target_quantity"] * 2
    )
    assert (
        large_positions["MSFT"]["target_quantity"]
        == small_positions["MSFT"]["target_quantity"] * 2
    )


def test_whole_share_policy_rounds_targets_and_keeps_residual_cash(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    fractional_registry = _write_registry(
        tmp_path / "fractional.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
        quantity_policy="fractional",
    )
    whole_share_registry = _write_registry(
        tmp_path / "whole.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
        quantity_policy="whole_share",
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    fractional = portfolio_status(
        state_root, fractional_registry, "p1", as_of=date(2026, 1, 1)
    )
    whole = portfolio_status(
        state_root, whole_share_registry, "p1", as_of=date(2026, 1, 1)
    )
    fractional_positions = {
        row["asset"]: row for row in fractional["target_snapshot"]["positions"]
    }
    whole_positions = {
        row["asset"]: row for row in whole["target_snapshot"]["positions"]
    }

    assert fractional_positions["MSFT"]["target_quantity"] == 2.5
    assert whole_positions["MSFT"]["target_quantity"] == 2.0
    assert whole["target_snapshot"]["residual_cash"] == 100.0
    assert whole["rebalance_plan"]["summary"]["residual_cash"] == 100.0


def test_min_trade_threshold_suppresses_tiny_drift_but_keeps_row_visible(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
        top_n=1,
        gross_exposure=0.5,
        min_delta_quantity=0.02,
    )
    onboard_portfolio(
        state_root,
        "p1",
        starting_cash=501.0,
        imported_holdings=[{"symbol": "AAPL", "quantity": 4.99, "avg_cost": 100.0}],
    )

    status = portfolio_status(state_root, registry_path, "p1", as_of=date(2026, 1, 1))
    aapl = {row["asset"]: row for row in status["rebalance_plan"]["rows"]}["AAPL"]

    assert round(aapl["delta_quantity"], 6) == 0.01
    assert aapl["action"] == "no-op"
    assert aapl["suppressed_reason"] == "below_min_delta_quantity"
    assert status["rebalance_plan"]["instructions"] == []


def test_fill_journals_remain_isolated_per_portfolio(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)
    onboard_portfolio(state_root, "p2", starting_cash=1000.0)

    fill_one = record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 1),
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        fill_price=100.0,
        fill_id="fill-p1",
    )
    fill_two = record_fill(
        state_root,
        "p2",
        trade_date=date(2026, 1, 1),
        symbol="MSFT",
        side="buy",
        quantity=1.0,
        fill_price=200.0,
        fill_id="fill-p2",
    )

    p1_rows = read_jsonl(state_root / "p1" / "fills.jsonl")
    p2_rows = read_jsonl(state_root / "p2" / "fills.jsonl")
    assert [row["fill_id"] for row in p1_rows] == [fill_one["fill_id"]]
    assert [row["fill_id"] for row in p2_rows] == [fill_two["fill_id"]]


def test_validation_rejects_duplicate_imported_holdings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate holding symbol: AAPL"):
        onboard_portfolio(
            tmp_path / "state",
            "p1",
            starting_cash=1000.0,
            imported_holdings=[
                {"symbol": "AAPL", "quantity": 1.0, "avg_cost": 100.0},
                {"symbol": "aapl", "quantity": 1.0, "avg_cost": 101.0},
            ],
        )


def test_validation_rejects_bad_fill_inputs(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    with pytest.raises(ValueError, match="commission must be non-negative"):
        record_fill(
            state_root,
            "p1",
            trade_date=date(2026, 1, 1),
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            fill_price=100.0,
            commission=-0.01,
        )


def test_invalid_registry_mapping_blocks_daily_run_cleanly(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    payload = yaml.safe_load(registry_path.read_text())
    payload["artifacts"]["signal"]["value_col"] = "missing_signal"
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 1))
    portfolio = result["portfolios"]["p1"]

    assert result["run_status"] == "blocked"
    assert portfolio["run_status"] == "blocked"
    assert (
        "signal artifact missing required columns" in portfolio["blocking_reasons"][0]
    )


def test_missing_signal_price_overlap_blocks_with_clear_error(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path = tmp_path / "signals.parquet"
    price_path = tmp_path / "prices.parquet"
    pl.DataFrame(
        {"timestamp": [date(2026, 1, 1)], "asset": ["AAPL"], "signal": [1.0]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime)).write_parquet(signal_path)
    pl.DataFrame(
        {"date": [date(2026, 1, 1)], "asset": ["MSFT"], "price": [200.0]}
    ).with_columns(pl.col("date").cast(pl.Datetime)).write_parquet(price_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 1))

    assert result["portfolios"]["p1"]["run_status"] == "blocked"
    assert (
        "no overlapping signal and price rows"
        in result["portfolios"]["p1"]["blocking_reasons"][0]
    )


def test_daily_run_isolates_portfolio_failures_and_preserves_good_output(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    result = daily_run(
        state_root,
        registry_path,
        as_of=date(2026, 1, 1),
        portfolio_ids=["p1", "missing"],
    )

    assert result["run_status"] == "warning"
    assert result["portfolios"]["p1"]["run_status"] == "ok"
    assert result["portfolios"]["missing"]["run_status"] == "blocked"
    assert (state_root / "p1" / "daily" / "2026-01-01" / "daily_run.json").exists()
    assert (state_root / "missing" / "daily" / "2026-01-01" / "daily_run.json").exists()


def test_strategy_switching_via_manual_promotion_file(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    status_alpha = portfolio_status(
        state_root, registry_path, "p1", as_of=date(2026, 1, 1)
    )
    alpha_assets = [
        row["asset"] for row in status_alpha["target_snapshot"]["positions"]
    ]
    assert alpha_assets == ["AAPL", "MSFT"]

    alt_signal_path = tmp_path / "signals_b.parquet"
    pl.DataFrame(
        {
            "timestamp": [date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 1)],
            "asset": ["AAPL", "MSFT", "NVDA"],
            "signal": [1.0, 5.0, 6.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime)).write_parquet(alt_signal_path)
    _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="beta",
        signal_path=alt_signal_path,
        price_path=price_path,
    )

    status_beta = portfolio_status(
        state_root, registry_path, "p1", as_of=date(2026, 1, 1)
    )
    beta_assets = [row["asset"] for row in status_beta["target_snapshot"]["positions"]]
    assert status_beta["strategy_id"] == "beta"
    assert beta_assets == ["NVDA", "MSFT"]


def test_write_json_is_atomic_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    payload = {"z": 1, "a": {"d": 4, "c": 3}}

    write_json(path, payload)
    write_json(path, payload)

    assert read_json(path) == payload
    assert (
        path.read_text() == '{\n  "a": {\n    "c": 3,\n    "d": 4\n  },\n  "z": 1\n}\n'
    )
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_append_jsonl_and_tail_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "fills.jsonl"

    append_jsonl(path, {"b": 2, "a": 1})
    append_jsonl(path, {"b": 4, "a": 3})
    append_jsonl(path, {"b": 6, "a": 5})

    assert path.read_text().splitlines() == [
        '{"a":1,"b":2}',
        '{"a":3,"b":4}',
        '{"a":5,"b":6}',
    ]
    assert read_jsonl(path) == [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
    assert tail_jsonl(path, 2) == [{"a": 3, "b": 4}, {"a": 5, "b": 6}]


def test_daily_run_repeated_writes_leave_stable_outputs(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)

    first = daily_run(state_root, registry_path, as_of=date(2026, 1, 1))
    second = daily_run(state_root, registry_path, as_of=date(2026, 1, 1))

    output_dir = state_root / "p1" / "daily" / "2026-01-01"
    assert (
        first["portfolios"]["p1"]["target_snapshot"]
        == second["portfolios"]["p1"]["target_snapshot"]
    )
    assert (
        json.loads((output_dir / "daily_run.json").read_text())["portfolio_id"] == "p1"
    )
    assert not list(output_dir.glob("*.tmp"))


def test_manual_workflow_has_no_live_runtime_dependencies() -> None:
    package_root = Path(__file__).resolve().parents[1] / "manual_portfolio"
    forbidden_modules = {"ml4t.live", "live"}
    forbidden_names = {"RiskState", "SafeBroker", "LiveEngine", "VirtualPortfolio"}

    for path in sorted(package_root.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules
                    assert alias.name not in forbidden_modules
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.split(".")[0] not in forbidden_modules
                assert node.module not in forbidden_modules
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert forbidden_names.isdisjoint(names)
