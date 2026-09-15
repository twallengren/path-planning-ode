#!/usr/bin/env python3
"""Run one small, reproducible terrain-planning example."""

from path_planning_ode import PlannerConfig, evaluate_route, plan, validation_terrain


def main() -> None:
    scenario = validation_terrain("uniform")
    config = PlannerConfig(
        method="slsqp",
        initialization="straight",
        interior_points=8,
        reference_grid_size=33,
        max_iterations=100,
        time_limit_s=10.0,
        profile_samples=33,
    )
    result = plan(scenario, config)
    print(f"{scenario.name}: {result.termination_reason}")
    if result.route_m is None:
        raise SystemExit("planner returned no route")
    evaluation = evaluate_route(scenario, result.route_m, profile_samples=33)
    print(
        f"feasible={evaluation.feasible} cost_s={evaluation.cost_s:.3f} "
        f"length_m={evaluation.length_m:.3f}"
    )


if __name__ == "__main__":
    main()
