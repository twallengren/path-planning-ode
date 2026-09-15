# Playground numerical audit

This audit checks the interactive solver against definitions that do not use
its optimization implementation. The audit suite is
`tests/test_playground_audit.py`.

## Quantities checked

The displayed polygonal path uses an equal parameter interval for each
segment. Its auxiliary energy is

\[
E(q)=\int_0^1 c(q(t))^2\lVert q'(t)\rVert^2\,dt.
\]

The audit integrates this expression segment by segment with SciPy's adaptive
quadrature. It supplies each Gaussian's closest point on a segment as an
integration breakpoint, which prevents an independently written oracle from
missing a very narrow bump. A width-0.035 Gaussian on a highly uneven path
agrees with the playground's composite Gauss--Legendre result to the audit
tolerance. A separate central finite-difference calculation checks the
analytic gradient while holding the quadrature panels fixed.

Travel cost is checked separately against the legacy analytic Gaussian line
integral. Free-gradient RMS excludes both endpoints and an optional held
vertex. ODE-residual RMS uses the corresponding free rows of the legacy
Euler--Lagrange residual. These are distinct diagnostics: auxiliary energy is
the descent objective, travel cost measures the route, and the ODE residual
measures stationary-equation error.

## Editing, constraints, and line search

The audit stretches a path segment across a narrow Gaussian and verifies that
the new state allocates more quadrature panels and still matches adaptive
integration. For each descent step, the implementation reserves panels for
the largest permitted trial and uses that common quadrature for the baseline,
gradient, and Armijo trials. Independent adaptive energies are nonincreasing
across accepted descent steps when the field and fixed vertices remain
unchanged.

Endpoints and the held vertex remain exact. Releasing a held vertex preserves
the accepted geometry before making that coordinate free again. The Newton
audit independently removes the held vertex's two rows and columns from the
legacy ODE Jacobian and confirms that the accepted update is a scalar-damped
version of that reduced solve.

## Resolution evidence

On a smooth off-centre Gaussian fixture, Newton converges in five updates at
32, 64, and 128 interior points. The relative variation of midpoint weighted
speed decreases from approximately `1.23e-4` to `3.18e-5` to `8.08e-6`, and
the 64-to-128 path change is smaller than the 32-to-64 change.

The energy-descent method also converges at all three resolutions using its
scaled free-gradient criterion. Its independently reported ODE-residual RMS
decreases from approximately `1.38e-2` to `3.59e-3` to `8.97e-4`. Thus the
descent stationary paths become consistent with the continuous ODE under mesh
refinement even though descent does not solve the ODE linear system directly.

A symmetric central-hill case produces upper, lower, and direct stationary
candidates. The reflected detours agree, while the direct stationary route's
travel cost is much larger. This is a deliberate check that stationarity is
not treated as a global-optimality certificate.

## Brush and budget checks

Sparse and densely sampled versions of the same stroke generate identical
Gaussian fields. A click's centre cost equals its nominal multiplicative brush
strength, and the interior of a long stroke approaches the same value. The
256-Gaussian cap raises an error before allocating an oversized generated
field; it does not thin the stroke silently. The solver enforces a hard maximum
of 2,000 iterations.

## Audit result

All independent numerical checks pass. The claims above apply to the tested
discrete objectives and smooth refinement fixture. They do not certify a
global minimum or convergence for every user-drawn field.

The built Pyodide worker was also timed headlessly on an x86-64 Mac running
macOS 15.7.9 and Chrome for Testing 153.0.8010.12. At 32 interior points,
default-field step round trips had an 18.5 ms median and 24.9 ms maximum over
61 updates; after painting a narrow field they had a 45.6 ms median and 58.4
ms maximum over 40 updates. A 390 by 844 mobile viewport on the same desktop
machine measured 20.3/25.3 ms and 46.1/49.5 ms median/maximum respectively.
That viewport run checks responsive layout and browser scheduling; it is not a
measurement on mobile hardware. The default converged browser path agreed
with native Python to `1.11e-15` maximum coordinate error.
