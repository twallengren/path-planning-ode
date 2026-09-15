# Weighted distance: mathematical conventions

For a typeset, step-by-step derivation, read
[From a cost to a curve](https://twallengren.github.io/path-planning-ode/derivation.html)
([HTML source](../web/derivation.html)). This is the compact reference for the solver.

## Objective

Fix q(0) = a and q(1) = b. The parameter t labels the curve; it is not physical time.

$$J[q]=\int_0^1 c(q)\|q'\|\,dt,\qquad c(q)=1+\sum_i w_i e^{-\|q-o_i\|^2/\sigma_i^2}.$$

Weights are nonnegative and widths positive, so c ≥ 1. Width is the 1/e radius.
Hills are soft costs, not collision boundaries. Weighted distance is invariant
under orientation-preserving reparameterization.

For a smooth regular curve, direct Euler–Lagrange gives

$$\frac{d}{dt}\left(\frac{c q'}{\|q'\|}\right)-\|q'\|\nabla c=0.$$

This determines bending but leaves the traversal schedule free.

## Parameterization and ODE

Use the auxiliary energy E = ∫ c²‖q′‖² dt. Cauchy–Schwarz gives J² ≤ E on [0,1],
with equality at constant weighted speed c‖q′‖. Every regular geometric route
admits that parameterization. Minimizing E over curves and parameterizations
therefore gives the same minimizing geometric routes as minimizing J.
This does not equate the objectives for an arbitrary fixed schedule.

Euler–Lagrange for E gives, with v = q′,

$$q''=F(q,v)=\frac{\|v\|^2\nabla c-2v(\nabla c\cdot v)}{c}.$$

$$x''=\frac{c_x(y'^2-x'^2)-2c_yx'y'}{c},\qquad
y''=\frac{c_y(x'^2-y'^2)-2c_xx'y'}{c}.$$

Along a continuous solution, c²‖v‖² is constant. Stationarity does not prove
minimality. Coincident endpoints admit the constant zero-cost path.

## Discretization and Jacobian

N interior points means N+2 stored points and N+1 intervals: h = 1/(N+1).
Unknowns are interleaved [x₁, y₁, x₂, y₂, …]; endpoints are fixed.

$$R_i=\frac{q_{i+1}-2q_i+q_{i-1}}{h^2}-F\left(q_i,\frac{q_{i+1}-q_{i-1}}{2h}\right).$$

For g = ∇c and H = ∇²c, the analytic derivatives are

$$F_q=\frac{\|v\|^2H-2v(Hv)^T}{c}-\frac{Fg^T}{c},\qquad
F_v=\frac{2(gv^T-vg^T-(g\cdot v)I)}{c}.$$

$$J_{i,i-1}=I/h^2+F_v/(2h),\quad
J_{i,i}=-2I/h^2-F_q,\quad J_{i,i+1}=I/h^2-F_v/(2h).$$

The implementation uses a dense linear solve. This second-order finite-difference
system approximates the continuous ODE, not the exact gradient of the polyline cost.

Starting curves are qβ(t) = a + t(b−a) + 0.3β sin(πt) n, where
n = (−(bᵧ−aᵧ), bₓ−aₓ). Direct uses β=0, Right arc β=−1, Left arc β=1.
Their JSON keys are straight, bend-x, and bend-y.

## Iteration and stopping

Solve JΔ = −R; update interior coordinates by αΔ. Damped mode tries
α = 1, 1/2, …, 2⁻²⁰ and accepts the first finite trial satisfying

$$\|R_{\mathrm{new}}\|^2\le(1-10^{-4}\alpha)\|R_{\mathrm{old}}\|^2.$$

Undamped mode takes a full finite step. The stopping norm is RMS = ‖R‖/√(2N).
Defaults: tolerance 1e−7, 100 iterations.

Statuses: running, converged, stagnated (no acceptable backtracking step),
singular (failed linear solve), nonfinite, or iteration_limit.
Failed updates retain the last accepted path and diagnostics.

## Measure the route

The weighted_distance function integrates c ds along each polyline segment.
For one Gaussian bump and a segment of length ℓ and unit tangent T starting at p,
set a = (p−o)·T, and let ρ be the perpendicular distance to its supporting line.
Its contribution is

$$w e^{-\rho^2/\sigma^2}\frac{\sigma\sqrt{\pi}}{2}
\left[\operatorname{erf}\left(\frac{a+\ell}{\sigma}\right)
-\operatorname{erf}\left(\frac{a}{\sigma}\right)\right].$$

Add all bump contributions and baseline length ℓ. The implementation uses erfc
in Gaussian tails to reduce cancellation. Zero-length segments contribute zero.
Subdivision of a straight segment, reversed traversal, and repeated vertices do
not change the result, apart from floating-point rounding.

The state reports cost (weighted distance) and length (geometric polyline length).
Energy remains an auxiliary midpoint estimate of ∫ c²‖q′‖² dt.
Neither cost nor energy is the Newton damping merit function.

“Lowest cost shown” compares displayed candidates, including unfinished ones;
it is not a global optimality claim. A low residual can describe a high-cost
stationary curve. Exact polyline cost integration does not eliminate ODE mesh
error: refine the mesh and compare starting routes.
