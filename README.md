# The shape of a path

An interactive case study of path planning with the Euler–Lagrange equations.
Move obstacles, compare three initial guesses, and watch Newton's method reshape
a path through a landscape of soft costs.

**[Explore the website](https://twallengren.github.io/path-planning-ode/)** ·
**[Read the full derivation](https://twallengren.github.io/path-planning-ode/derivation.html)** ·
**[Open the notebook](examples/case_study.ipynb)**

The same Python solver runs locally and in the browser through Pyodide. The site
is static: computation happens on the visitor's device, with no API keys or server.

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

Export a JSON scene from the website and reproduce it:

```sh
uv run --extra plot python examples/explore.py path-scene.json
```

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
    print(guess, final.status, final.residual_norm, final.energy)
    # final.path has shape (N + 2, 2), including endpoints.
```

Use `initialize(scene, guess)` and `step(scene, state)` for single iterations.
`result.histories` contains snapshots for each guess, with the path, iteration,
RMS residual, midpoint energy, geometric length, damping factor, and status.
A terminal state is returned unchanged by `step`.

`Scene.to_dict()` and `Scene.from_dict(data)` provide version 1 JSON interchange.
The guesses are `straight`, `bend-x` (t⁵, t), and `bend-y` (t, t⁵). `presets()`
returns deterministic scenes: `empty`, `central`, `asymmetric`, `passage`, and
`challenge`. No random seed is required.

## Explore the notebook

```sh
uv sync --extra notebook
uv run jupyter lab examples/case_study.ipynb
```

The notebook explains the objective, compares starting guesses and Newton modes,
plots convergence, and demonstrates scene interchange.

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
essay appears before Python is ready, with a labeled precomputed example and
retry controls on loading failure.

Browser limits: 1–100 interior points, 20 obstacles, 100 iterations, coordinates
in [−100, 100], obstacle weights in [0, 100], widths in [0.1, 10], and tolerance
in [1e−12, 1e−2]. The Python API supports larger experiments. Heatmap colors are
relative within each scene; the solver uses actual costs.

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

The objective is E[q] = ∫ c(q)‖q′‖² dt, with fixed endpoints. Each obstacle adds
a Gaussian bump to the positive cost field. Euler–Lagrange gives a coupled
second-order ODE. Central differences turn it into a nonlinear system, solved
with damped or undamped Newton iterations.

- Obstacles are soft costs, not hard collision constraints.
- A small residual does not establish a minimum or global optimality.
- Damping seeks residual decrease, not energy decrease. Initial guesses and mesh
  resolution influence results; narrow obstacles can be undersampled.
- Rover playback uses constant geometric speed, not simulated robot dynamics.

See [the mathematical conventions](docs/mathematics.md).

## From the 2019 version

The original project linked to [this derivation video](https://www.youtube.com/watch?v=fNBrIngCJp8).
The script is preserved in [legacy/pathplanning2019.py](legacy/pathplanning2019.py).
It has known numerical inconsistencies and is not the supported entry point.
The old import/API is not retained: use the examples above to migrate. The new
solver corrects the second derivative scale and timestep, verifies the Jacobian,
replaces inversion with a linear solve, and reports failures explicitly. Both
browser modes use the corrected equations.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project layout and change guidelines.
