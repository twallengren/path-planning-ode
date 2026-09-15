# Working on this project

- `src/path_planning_ode/`: solver, interchange objects, deterministic presets.
- `web/src/`: essay, controls, rendering, Pyodide module worker.
- `tests/` and `web/tests/`: numerical and browser regression tests.
- `examples/`: CLI plotting example and executable notebook.
- `scripts/`: asset preparation and the test server.
- `legacy/`: historical source, excluded from modern checks.

Run the README's checks before submitting changes. Commit both lockfiles when
changing dependencies. Generated runtime files and build output are ignored.

Keep numerical behavior in Python; the worker imports that package directly.
Add numerical regression tests for solver changes. Rebuild assets after Python
changes. Preserve scene versioning, documented browser limits, and native/browser
parity. SymPy and Matplotlib should remain optional outside the browser runtime.

For UI changes, check the production build at the Pages prefix, keyboard and
pointer input, a narrow screen, and reduced motion. For numerical changes, inspect
residual behavior and mesh convergence, not only screenshots.
