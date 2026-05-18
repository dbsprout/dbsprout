"""Tests for dbsprout.migrate.update_models — update action data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.migrate.update_models import (
    PlannedAction,
    UpdateAction,
    UpdatePlan,
    UpdateResult,
)

# ── UpdateAction enum tests ────────────────────────────────────────────────

_EXPECTED_ACTIONS = frozenset(
    {
        "GENERATE_COLUMN",
        "DROP_COLUMN",
        "GENERATE_TABLE",
        "DROP_TABLE",
        "REGENERATE_COLUMN",
        "VALIDATE_FK",
        "DEDUPLICATE",
        "REPLACE_ENUM",
        "NO_ACTION",
    }
)


class TestUpdateAction:
    def test_has_all_9_variants(self) -> None:
        actual = {m.name for m in UpdateAction}
        assert actual == _EXPECTED_ACTIONS

    def test_exactly_9_variants(self) -> None:
        assert len(UpdateAction) == 9

    def test_values_are_lowercase_snake_case(self) -> None:
        for member in UpdateAction:
            assert member.value == member.name.lower()

    def test_is_str_enum(self) -> None:
        assert isinstance(UpdateAction.GENERATE_COLUMN, str)
        assert UpdateAction.GENERATE_COLUMN == "generate_column"

    def test_all_values_are_strings(self) -> None:
        for member in UpdateAction:
            assert isinstance(member.value, str)
            assert isinstance(member, str)


# ── Helper fixtures ────────────────────────────────────────────────────────


def _make_change(
    change_type: SchemaChangeType = SchemaChangeType.COLUMN_ADDED,
    table_name: str = "users",
    column_name: str | None = "email",
) -> SchemaChange:
    return SchemaChange(
        change_type=change_type,
        table_name=table_name,
        column_name=column_name,
    )


def _make_planned_action(
    change: SchemaChange | None = None,
    action: UpdateAction = UpdateAction.GENERATE_COLUMN,
    description: str = "Generate data for new column",
) -> PlannedAction:
    return PlannedAction(
        change=change or _make_change(),
        action=action,
        description=description,
    )


# ── PlannedAction model tests ──────────────────────────────────────────────


class TestPlannedAction:
    def test_required_fields(self) -> None:
        change = _make_change()
        pa = PlannedAction(
            change=change,
            action=UpdateAction.GENERATE_COLUMN,
            description="Generate data for new column",
        )
        assert pa.change == change
        assert pa.action == UpdateAction.GENERATE_COLUMN
        assert pa.description == "Generate data for new column"

    def test_frozen_model(self) -> None:
        pa = _make_planned_action()
        with pytest.raises(ValidationError):
            pa.description = "mutated"  # type: ignore[misc]

    def test_frozen_change_field(self) -> None:
        pa = _make_planned_action()
        with pytest.raises(ValidationError):
            pa.change = _make_change(table_name="orders")  # type: ignore[misc]

    def test_frozen_action_field(self) -> None:
        pa = _make_planned_action()
        with pytest.raises(ValidationError):
            pa.action = UpdateAction.DROP_COLUMN  # type: ignore[misc]

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannedAction(  # type: ignore[call-arg]
                change=_make_change(),
                action=UpdateAction.GENERATE_COLUMN,
            )
        with pytest.raises(ValidationError):
            PlannedAction(  # type: ignore[call-arg]
                change=_make_change(),
                description="desc",
            )
        with pytest.raises(ValidationError):
            PlannedAction(  # type: ignore[call-arg]
                action=UpdateAction.GENERATE_COLUMN,
                description="desc",
            )

    def test_each_action_type_accepted(self) -> None:
        for action in UpdateAction:
            pa = PlannedAction(
                change=_make_change(),
                action=action,
                description=f"Action: {action.value}",
            )
            assert pa.action == action


# ── UpdatePlan model tests ─────────────────────────────────────────────────


class TestUpdatePlan:
    def test_default_empty_actions(self) -> None:
        plan = UpdatePlan()
        assert plan.actions == []

    def test_with_actions(self) -> None:
        pa1 = _make_planned_action(description="first")
        pa2 = _make_planned_action(
            action=UpdateAction.DROP_TABLE,
            description="second",
        )
        plan = UpdatePlan(actions=[pa1, pa2])
        assert len(plan.actions) == 2
        assert plan.actions[0].description == "first"
        assert plan.actions[1].description == "second"

    def test_frozen_model(self) -> None:
        plan = UpdatePlan(actions=[_make_planned_action()])
        with pytest.raises(ValidationError):
            plan.actions = []  # type: ignore[misc]


# ── UpdateResult model tests ───────────────────────────────────────────────


class TestUpdateResult:
    def test_required_fields(self) -> None:
        result = UpdateResult(
            tables_data={"users": [{"id": 1}]},
            actions_applied=[_make_planned_action()],
        )
        assert result.tables_data == {"users": [{"id": 1}]}
        assert len(result.actions_applied) == 1

    def test_default_values(self) -> None:
        result = UpdateResult(
            tables_data={},
            actions_applied=[],
        )
        assert result.rows_modified == 0
        assert result.rows_added == 0
        assert result.rows_removed == 0
        assert result.tables_added == []
        assert result.tables_removed == []

    def test_all_fields_populated(self) -> None:
        pa = _make_planned_action()
        result = UpdateResult(
            tables_data={"users": [{"id": 1, "name": "Alice"}]},
            actions_applied=[pa],
            rows_modified=5,
            rows_added=10,
            rows_removed=2,
            tables_added=["orders"],
            tables_removed=["legacy"],
        )
        assert result.rows_modified == 5
        assert result.rows_added == 10
        assert result.rows_removed == 2
        assert result.tables_added == ["orders"]
        assert result.tables_removed == ["legacy"]

    def test_frozen_model(self) -> None:
        result = UpdateResult(
            tables_data={},
            actions_applied=[],
        )
        with pytest.raises(ValidationError):
            result.rows_modified = 99  # type: ignore[misc]

    def test_frozen_tables_data(self) -> None:
        result = UpdateResult(
            tables_data={"users": [{"id": 1}]},
            actions_applied=[],
        )
        with pytest.raises(ValidationError):
            result.tables_data = {}  # type: ignore[misc]

    def test_frozen_tables_added(self) -> None:
        result = UpdateResult(
            tables_data={},
            actions_applied=[],
            tables_added=["t1"],
        )
        with pytest.raises(ValidationError):
            result.tables_added = []  # type: ignore[misc]

    def test_frozen_tables_removed(self) -> None:
        result = UpdateResult(
            tables_data={},
            actions_applied=[],
            tables_removed=["t1"],
        )
        with pytest.raises(ValidationError):
            result.tables_removed = []  # type: ignore[misc]

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            UpdateResult(  # type: ignore[call-arg]
                actions_applied=[],
            )
        with pytest.raises(ValidationError):
            UpdateResult(  # type: ignore[call-arg]
                tables_data={},
            )
