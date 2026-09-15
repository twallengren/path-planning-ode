# Mathematics and numerical conventions

## Objective and cost

Let q(t) = (x(t), y(t)), t ∈ [0, 1], with prescribed endpoints. Define

$$c(q)=1+\sum_i w_i\exp(-\|q-o_i\|^2/s_i^2),\qquad E[q]=\int_0^1 c(q)\|q'\|^2\,dt.$$

Weights are nonnegative and widths positive, so c ≥ 1. Width one recovers the
original Gaussian. Width is the 1/e radius, not the standard deviation of a
normalized Gaussian. No obstacle is a forbidden region.

This is parameterized energy, not ordinary path length. For a fixed geometric
curve, minimizing over parameterizations makes √c‖q′‖ constant and relates energy
to the square of weighted length ∫√c ds. The implementation solves the ODE on a
uniform parameter grid; it does not explicitly reparameterize curves.

## Euler–Lagrange equations

For L = c(q)‖q′‖², ∂L/∂q′ = 2c q′ and ∂L/∂q = ∇c‖q′‖². Thus

$$2c q''+2(\nabla c\cdot q')q'-\nabla c\|q'\|^2=0.$$

Writing v = q′,

$$q''=F(q,v)=\frac{\|v\|^2\nabla c-2v(\nabla c\cdot v)}{2c}.$$

In coordinates this recovers the original equations:

$$x''=\frac{c_x(y'^2-x'^2)-2c_yx'y'}{2c},\qquad y''=\frac{c_y(x'^2-y'^2)-2c_xx'y'}{2c}.$$

The symbolic test independently constructs the Euler–Lagrange equations and
compares their solution with the implementation. Stationarity does not establish
local or global minimality.

## Discretization and Jacobian

N means **interior points**: N+2 stored points and N+1 intervals, h = 1/(N+1).
Endpoints never enter the Newton unknown vector. Interior coordinates are
interleaved: [x₁, y₁, x₂, y₂, …].

$$R_i=\frac{q_{i+1}-2q_i+q_{i-1}}{h^2}-F\left(q_i,\frac{q_{i+1}-q_{i-1}}{2h}\right).$$

Let g = ∇c and H = ∇²c. The analytic derivatives are

$$F_q=\frac{\|v\|^2H-2v(Hv)^T}{2c}-\frac{Fg^T}{c},\qquad F_v=\frac{gv^T-vg^T-(g\cdot v)I}{c}.$$

Nonzero Jacobian blocks in each block row are

$$J_{i,i-1}=I/h^2+F_v/(2h),\quad J_{i,i}=-2I/h^2-F_q,\quad J_{i,i+1}=I/h^2-F_v/(2h).$$

A dense linear solve is sufficient for the browser's small systems. The residual
approximates the continuous ODE; it is not exactly the gradient of the separately
displayed midpoint energy. Mesh refinement tests check consistency.

## Newton and stopping

Solve JΔ = −R and update interior points by αΔ. Undamped mode uses α=1. Damped
mode tries α=1, 1/2, …, 2⁻²⁰, accepting the first finite trial satisfying

$$\|R_{new}\|^2\le(1-10^{-4}\alpha)\|R_{old}\|^2.$$

The displayed norm is RMS = ‖R‖/√(2N). Defaults: tolerance 1e−7, 100 iterations.
Outcomes are `running`, `converged`, `stagnated` (no acceptable backtracking step),
`singular` (linear solve failed), `nonfinite`, or `iteration_limit`. Failed steps
retain the last finite path and diagnostics. Initially converged guesses have
zero iterations. Convergence takes precedence over the iteration limit.

Energy uses midpoint quadrature over polyline segments; length is the sum of
Euclidean segment lengths. Neither is used as the damping merit function.
Residual convergence implies neither monotone energy nor clearance.

## Experiments

1. Empty fields recover a straight line in at most one step from all three
   polynomial guesses. Energy is squared endpoint distance.
2. Central symmetry can preserve a straight-through stationary path, even through
   high cost. Bent guesses expose other stationary paths.
3. Refine the mesh around narrow bumps. A coarse low residual is not evidence of
   an accurate continuous solution.
4. Reset before comparing damped and undamped modes on the challenging preset.

The original linked video is credited in the README; this derivation was
recovered from and independently checked against the source equations.
