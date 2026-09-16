# Unified solver notes

The playground uses one field and solver interface for analytic Gaussian maps
and sampled terrain maps. The consolidated implementation is independently
audited in `tests/test_unified_audit.py`; those checks directly exercise
`field_adapter.py` and `playground.py` and do not depend on earlier planner
modules or study artifacts.

## Field and objectives

A terrain-plus-brush field is the product of the terrain slowness and the
positive Gaussian factor. Its gradient and Hessian include both product-rule
cross terms. The audit compares these derivatives with centered numerical
differences and compares displayed travel cost, \(\int c\,ds\), with adaptive
quadrature to relative tolerance \(10^{-5}\), including a narrow Gaussian on a
terrain grid.

The polygonal path has one fixed parameter interval per segment. Descent uses
the quadrature approximation of

\[
E[q]=\int_0^1 c(q)^2\lvert q'\rvert^2\,dt
\]

and the exact gradient of that same fixed-panel approximation. Travel cost,
energy-gradient RMS, and ODE-residual RMS remain separate diagnostics.

Version 0.3 hashes use browser-stable JSON number semantics: an integral float
and the corresponding integer hash identically, as do positive and negative
zero. This prevents an unchanged Python document from failing validation after
a JavaScript JSON round trip. Non-integral changes still alter the hash, and a
terrain field specification still rejects a supplied scenario hash that does
not match its data. Hashes from older spelling-sensitive versions can differ.

For optimization, \(z=q/L\) and
\(F=E/(L c_\mathrm{ref})^2\). The length scale \(L\) is the field-bounds
diagonal, or the initial path bounding-box diagonal with a floor of one for an
unbounded field. It is frozen for the solve. The audit scales coordinates by
1,000 and slowness inversely, then verifies identical normalized diagnostics,
line-search steps, and physical paths after converting units back.

## Automatic strategy

Automatic mode first applies stiffness-preconditioned energy descent with
Armijo backtracking. After at least eight accepted descent updates, recent
energy gains or the gradient threshold can schedule a Newton handoff. A
stationary or failed descent can request an immediate guarded Newton trial
before that threshold. Newton steps reduce the discrete ODE residual and must
stay within the domain and the move cap. Across the whole Newton phase they
must also stay within 0.1% of the handoff travel cost and 1% of the handoff
energy.

If a Newton phase is rejected, descent resumes with a cooldown of eight
**accepted** updates. Rejected descent attempts do not consume cooldown. A
descent failure during cooldown returns `backtracking_failed` with the last
accepted path. Automatic convergence requires the scaled free ODE-residual RMS
to be at most \(10^{-5}\); a small energy gradient alone cannot report automatic
convergence.

Endpoints and an optional held pin are removed from both reduced systems. A
pin therefore imposes piecewise stationarity on the free vertices on either
side. Coincident endpoints and looped paths are supported. A domain-boundary
path can terminate with a structured backtracking failure when every descent
trial points outside; the solver retains the valid submitted path.

## Audit evidence

Run the independent checks with:

```bash
.venv/bin/pytest -q tests/test_unified_audit.py
```

The audit includes real Gaussian and terrain solves rather than only mocked
phase transitions. On the audit machine:

- the supplied hard wobble converged at 32, 64, and 128 interior points in 14
  accepted-or-terminal updates, with independently evaluated travel costs
  8.392554, 8.392267, and 8.392194;
- the smooth 32-point fixture handed off after descent and reached a scaled ODE
  residual of about \(1.6\times10^{-8}\) in 13 updates;
- the audited 32-point terrain-plus-Gaussian case converged in 14 updates, with
  a median native update time of about 9 ms in an informal local timing run;
- a real Newton proposal rejected by the phase-wide guards entered cooldown,
  and exactly eight accepted descent updates cleared it.

These are focused numerical checks and timing observations. They do not show a
global optimum, and timings depend on the machine. Mesh refinement remains the
way to assess the gap between the discrete path equations and the continuous
ODE.
