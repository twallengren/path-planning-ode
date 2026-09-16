# Path planning ODE

An interactive playground for routes through a landscape with a cost at every point. Paint hills or walls, reshape a route, and watch the path adjust.

[Open the playground](https://twallengren.github.io/path-planning-ode/) · [Read the math](https://twallengren.github.io/path-planning-ode/derivation.html)

## Run it

Requires Python 3.11+, Node 24, [uv](https://docs.astral.sh/uv/), and npm.

```sh
uv sync
npm ci
npm run dev
```

Build the published site with `npm run build`. The browser runs the same Python package through Pyodide; there is no server-side solver.

## Python

The original Gaussian scene API remains available for experiments:

```python
from path_planning_ode import Obstacle, Scene, SolverOptions, solve

scene = Scene(start=(-2, -2), end=(12, 12), obstacles=(Obstacle(5, 5, 6, 2),), options=SolverOptions(interior_points=30))
result = solve(scene)
```

For editable and terrain-backed fields, use `path_planning_ode.playground` and `path_planning_ode.field_adapter`. Planner, fast-marching, benchmark, and study-report APIs were removed in 0.3.

```python
from path_planning_ode import PlaygroundField, build_playground_preset
from path_planning_ode.playground import advance_playground, initialize_playground

scene = build_playground_preset("random_hills", seed=0)
field = PlaygroundField(scene["base_field"], scene["gaussians"])
state = initialize_playground(scene["start"], scene["end"], field=field)
state = advance_playground(state, iterations=20)  # automatic descent/Newton controller
print(state.status, state.metrics.route_cost)
```

## Verify

```sh
uv run pytest
uv run ruff check src tests scripts
npm run format:check
npm run build
npm test
```

The bundled [Mount Tamalpais source data](src/path_planning_ode/data/mount_tamalpais.json) and its [provenance and attribution](src/path_planning_ode/data/mount_tamalpais.provenance.json) remain in the package.
