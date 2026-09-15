# The shape of a path

The shortest route isn't always the cheapest. Give every place a cost per unit
distance, then explore when a detour is worth taking. Move hills, vary their
strength and width, and compare candidate routes through the same landscape.

**[Open the path playground](https://twallengren.github.io/path-planning-ode/)** ·
**[Open the Gaussian explorer](https://twallengren.github.io/path-planning-ode/explorer.html)** ·
**[Read the full derivation](https://twallengren.github.io/path-planning-ode/derivation.html)** ·
**[Open the notebook](examples/case_study.ipynb)** ·
**[Open the terrain laboratory](https://twallengren.github.io/path-planning-ode/terrain.html)**

**[Terrain tutorial and reproducible study method](docs/terrain-study.md)** ·
**[Run the short terrain example](examples/terrain_experiment.py)**

The same Python solver runs locally and in the browser through Pyodide. The site
is static: computation happens on the visitor's device, with no API keys or server.

The homepage is a freehand playground: paint smooth cost regions, drag a route,
hold an interior waypoint, and watch incremental energy descent or ODE Newton
updates. It reports weighted route cost separately from the auxiliary energy,
free-gradient RMS, and ODE-residual RMS. The original form-based Gaussian
explorer remains available at `explorer.html`.

## Run locally

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/twallengren/path-planning-ode.git
cd path-planning-ode
uv sync --extra plot
uv run python examples/explore.py
```

`uv sync` creates a project `.venv` and installs the locked dependencies. Activation
is optional with `uv run`. For terminal output without Matplotlib:

```sh
uv sync
uv run python examples/explore.py --no-plot
```

Export a version-1 JSON scene from the **Gaussian explorer** and reproduce it:

```sh
uv run --extra plot python examples/explore.py path-scene.json
```

The homepage playground saves a different `kind: "path-playground"`, version-1
document containing its strokes, endpoints, current path, settings, and retained
routes. Load that document back into the playground. `examples/explore.py`
accepts Gaussian-explorer `Scene` files; it does not accept playground documents.

Prefer standard Python tools?

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot]'
python examples/explore.py
```

On Windows use `py -m venv .venv` and `.venv\Scripts\Activate.ps1`. The pip workflow
uses dependency ranges; uv uses the committed lockfile.

## Python API

```python
from path_planning_ode import Obstacle, Scene, SolverOptions, solve

scene = Scene(
    start=(-2, -2), end=(12, 12),
    obstacles=(Obstacle(x=5, y=5, weight=6, width=2),),
    options=SolverOptions(interior_points=30, mode="damped"),
)
result = solve(scene)
for guess, final in result.final.items():
    print(guess, final.status, final.cost, final.length)
    # final.path has shape (N + 2, 2), including endpoints.
```

Use `initialize(scene, guess)` and `step(scene, state)` for single iterations.
`result.histories` contains snapshots for each guess, with the path, iteration,
total weighted-distance `cost`, geometric `length`, RMS residual, damping factor,
and status. An auxiliary `energy` diagnostic estimates the integral of c² times
squared parameter speed; it is not the reported route cost.
A terminal state is returned unchanged by `step`.

`Scene.to_dict()` and `Scene.from_dict(data)` provide version 1 JSON interchange.
The guesses are `straight` (Direct), `bend-x` (Right arc), and `bend-y` (Left arc).
The arcs add a perpendicular sine displacement with peak 30% of the endpoint
distance, so they remain distinct for horizontal and vertical routes. `presets()`
returns deterministic scenes: `empty`, `central`, `asymmetric`, `passage`, and
`challenge`. No random seed is required.

For incremental editing, `path_planning_ode.playground` provides
`BrushStroke`, `PlaygroundOptions`, `initialize_playground`,
`advance_playground`, and `set_playground_pin`. A stroke's strength is its
nominal multiplicative cost: an isolated click reaches that value at its
centre, while a long stroke approaches it away from its ends. Paths retain
their submitted vertices; the incremental solver does not resample them.

```python
import numpy as np

from path_planning_ode import Obstacle
from path_planning_ode.playground import (
    PlaygroundOptions,
    advance_playground,
    initialize_playground,
)

path = np.column_stack((np.linspace(-4, 4, 34), np.linspace(0, 1, 34)))
state = initialize_playground(
    path[0],
    path[-1],
    (Obstacle(0, 0, weight=8, width=0.7),),
    path=path,
    options=PlaygroundOptions(interior_points=32, method="descent"),
    pin_index=16,
)
state = advance_playground(state, iterations=10)
print(state.status, state.metrics.route_cost, state.metrics.energy)
```

## Explore the notebook

```sh
uv sync --extra notebook
uv run jupyter lab examples/case_study.ipynb
```

The notebook compares route cost against distance, explores strength and width,
checks sampling invariance, and demonstrates scene interchange.

## Develop the website

Use Node.js 24 and npm alongside the Python environment:

```sh
uv sync
npm ci
npm run dev
```

Open the URL printed by Vite. The asset step copies the installed Python package,
generates example data, and prepares pinned Pyodide and NumPy assets. The first
build needs internet access; subsequent builds reuse downloaded runtime files.
Override `PYTHON` if your environment is outside `.venv`.

```sh
npm run build
npm run preview
```

Output is `web/dist/`. Runtime assets are served from the site, with no production
CDN dependency for computation. The runtime directory is about 16 MB before HTTP
compression. Fonts have system fallbacks if Google Fonts is unavailable. The
Gaussian explorer's essay appears before Python is ready, with a labelled
precomputed example and retry controls on loading failure. The playground canvas
remains editable if its local Python runtime fails to load.

Gaussian-explorer limits: 1–100 interior points, 20 obstacles, 100 iterations, coordinates
in [−100, 100], obstacle weights in [0, 100], widths in [0.1, 10], and tolerance
in [1e−12, 1e−2]. The Python API supports larger experiments. Heatmap colors are
relative within each scene; the solver uses actual costs.

The freehand playground offers 32, 64, or 128 interior points, retains at most
256 generated Gaussian bumps, and enforces a 2,000-iteration total budget.
See the [playground numerical audit](docs/playground-audit.md) for independent
quadrature, gradient, constraint, and mesh-refinement checks.

## Verify

```sh
uv run pytest
uv run ruff check src tests scripts examples
npm run format:check
npm run build
npx playwright install chromium
npm test
```

Tests cover the symbolic derivation, analytic Jacobian, fixed endpoints, empty
field, symmetry, mesh refinement, presets, and failure outcomes. Browser tests
compare the real worker with native Python and exercise editing, exports/imports,
sharing, playback, loading failure/retry, and mobile layout. They serve the built
site under `/path-planning-ode/` to catch Pages path problems. CI also executes
the notebook.

## GitHub Pages

The workflow checks pull requests and builds/deploys passing pushes to `master`
(the existing default branch). It supports manual dispatch too. In repository
**Settings → Pages**, select **GitHub Actions** as the source. The deployment
environment is `github-pages`.

Relative assets and fragment links let shared scenes refresh without SPA
rewrites. Forks can use the same workflow; update the essay's source links if
publishing under a different owner.

## What this algorithm does

The objective is total weighted distance: **add up local cost × distance traveled**
along a route with fixed endpoints. Traversing the same curve faster does not
change its cost. Each hill adds a Gaussian bump to a baseline cost of one.

To find stationary candidates, the methods use the equivalent continuous energy
with integrand c² times squared parameter speed. It selects constant weighted
speed; minimizing it over routes and their parameterizations gives the same
minimizing geometric routes. Euler–Lagrange gives a coupled second-order ODE.

The homepage playground keeps the submitted polyline vertices and defaults to
analytic-gradient descent on a composite-quadrature approximation of this
energy. Armijo backtracking accepts only energy-decreasing steps while the
field and fixed vertices stay unchanged. Its optional damped Newton method
instead reduces the centred-difference ODE residual; a temporarily held point
is removed from the free system until released.

The Gaussian explorer uses three fixed initial guesses and damped Newton to
solve the ODE boundary-value problem. Its iteration history and route
comparison are separate from the homepage's single editable path.

`weighted_distance(path, scene)` integrates the field analytically along every
segment of the displayed polyline, including narrow hills between vertices.
This is the `cost` used to measure routes in both browser tools. Subdividing a
straight segment leaves its cost unchanged. The Gaussian explorer also compares
against the direct endpoint-to-endpoint route and labels the cheapest candidate
**lowest cost shown**, not a global optimum.

- Obstacles are soft costs, not hard collision constraints.
- A small free-gradient or ODE-residual diagnostic does not establish a minimum
  or global optimality.
- Newton damping seeks residual decrease, not energy or route-cost decrease.
  Initial geometry and mesh resolution influence results; an ODE discretization
  can undersample narrow obstacles even when the route-cost diagnostic counts them.
- The Gaussian explorer's rover playback uses constant geometric speed, not
  simulated robot dynamics.

See [the mathematical conventions](docs/mathematics.md).

The v2 terrain pipeline uses shared SI-unit source fields, positive
log-slowness interpolation, and an independent route evaluator. The live
laboratory defaults to finite, smoothly transitioned high-cost walls at a
nominal 100× strength, so every planner may cross a wall and pays its field
cost. A selector retains the original impassable-barrier model for replay and
comparison. The checked-in publication remains explicitly the historical
hard-barrier study. See the [terrain tutorial](docs/terrain-study.md) for the
Python API, CLI protocol, browser demo, Mount Tamalpais provenance, and limits
on interpreting numerical references.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project layout and change guidelines.
