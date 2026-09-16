# Contributing

Keep numerical behavior in Python: the browser worker imports the packaged module directly. Add a focused numerical test for a solver or field change, then run `uv run pytest` and `npm run build`. Before `npm test`, install a browser with `npx playwright install chromium`.

`src/path_planning_ode/core.py` contains the supported Gaussian API. `terrain.py`, `terrain_generators.py`, and `field_adapter.py` provide terrain fields. `playground.py` contains the incremental route solver. The static site lives in `web/`.

The asset build clears and recopies `web/public/python/path_planning_ode`, so removed modules cannot be republished from a stale local build. Preserve the Mount Tamalpais data provenance and checksum when changing terrain assets.

The retired planner, fast-marching, and study interfaces are intentional breaking changes. Do not reintroduce reports or comparison dashboards without a user-facing need.
