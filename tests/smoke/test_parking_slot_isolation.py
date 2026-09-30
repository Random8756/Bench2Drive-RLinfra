from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTE_SCENARIO_PATHS = (
    Path("vendor/carla/training-runtime/leaderboard/scenarios/route_scenario.py"),
    Path("vendor/carla/training-runtime/leaderboard/scenarios/route_scenario_cloud.py"),
)


def _parking_catalog_copy_expression(path: Path) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    get_slots = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_parking_slots"
    )

    for node in ast.walk(get_slots):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "available_parking_locations"
            for target in node.targets
        ):
            continue

        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "list"
        ):
            return value

    raise AssertionError(f"No route-local parking catalog copy found in {path}")


def _parking_filter_loop(path: Path) -> tuple[ast.FunctionDef, ast.For]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    get_slots = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_parking_slots"
    )

    filter_loop = next(
        node
        for node in ast.walk(get_slots)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "slot"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "available_parking_locations"
    )
    return get_slots, filter_loop


@pytest.mark.parametrize("relative_path", ROUTE_SCENARIO_PATHS)
def test_parking_catalog_is_isolated_per_route_scenario(relative_path: Path) -> None:
    path = REPO_ROOT / relative_path
    expression = _parking_catalog_copy_expression(path)
    compiled = compile(ast.Expression(expression), str(path), "eval")

    first_slot = {"location": (1.0, 2.0, 3.0)}
    second_slot = {"location": (4.0, 5.0, 6.0)}
    module_catalog = [first_slot, second_slot]
    parked_vehicles = SimpleNamespace(Town12=module_catalog)
    namespace = {
        "__builtins__": {},
        "getattr": getattr,
        "list": list,
        "map_name": "Town12",
        "parked_vehicles": parked_vehicles,
    }

    first_episode_slots = eval(compiled, namespace)
    assert first_episode_slots == module_catalog
    assert first_episode_slots is not module_catalog
    assert first_episode_slots[0] is module_catalog[0]

    first_episode_slots.pop()
    second_episode_slots = eval(compiled, namespace)

    assert module_catalog == [first_slot, second_slot]
    assert second_episode_slots == module_catalog
    assert second_episode_slots is not first_episode_slots

    namespace["map_name"] = "Town01"
    missing_catalog_slots = eval(compiled, namespace)
    assert missing_catalog_slots == []
    assert missing_catalog_slots is not module_catalog


@pytest.mark.parametrize("relative_path", ROUTE_SCENARIO_PATHS)
def test_parking_filter_does_not_mutate_list_while_iterating(relative_path: Path) -> None:
    path = REPO_ROOT / relative_path
    get_slots, filter_loop = _parking_filter_loop(path)

    filtered_list_initializations = [
        node
        for node in ast.walk(get_slots)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "filtered_parking_locations"
            for target in node.targets
        )
        and isinstance(node.value, ast.List)
        and not node.value.elts
    ]
    assert len(filtered_list_initializations) == 1

    mutating_remove_calls = [
        node
        for node in ast.walk(filter_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remove"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "available_parking_locations"
    ]
    assert not mutating_remove_calls

    retention_conditions = [
        node
        for node in filter_loop.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and [
            value.id for value in node.test.values if isinstance(value, ast.Name)
        ] == ["in_area", "close_to_route"]
        and len(node.test.values) == 2
    ]
    assert len(retention_conditions) == 1

    retained_slot_appends = [
        node
        for node in ast.walk(retention_conditions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "filtered_parking_locations"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "slot"
    ]
    assert len(retained_slot_appends) == 1

    filtered_result_assignments = [
        node
        for node in ast.walk(get_slots)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "available_parking_locations"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "filtered_parking_locations"
    ]
    assert len(filtered_result_assignments) == 1
