# Method

## Task

Compile the propagator U = exp(-i H t) of an open, disordered mixed-field Ising chain,
H = sum_j J_j Z_j Z_{j+1} + sum_j h^z_j Z_j + sum_j h^x_j X_j, on n qubits (n = 4, 6, 8; times t = 0.5, 1.0, 1.5 in units of the mean coupling). Each instance draws J, h^z, h^x with 25% Gaussian multiplicative disorder around (1, 0.5, 1.05); a seed is one instance. Dense operators of size 2^n x 2^n are simulated exactly (torch, double precision, CPU).

## Circuits and the cost currency

A circuit is a sequence of layers: X (n rotations), D (n-1 two-qubit Z-Z rotations and n Z rotations, which commute and form one layer) and, for the generic baseline, RY, RZ and Clifford controlled-Z (CZ) layers. With K diagonal layers the Strang skeleton is X D X D ... D X; it has R = K(3n-1) + n non-Clifford rotations. The Suzuki fourth-order formula with r repeats uses the same skeleton with K = 5r and Suzuki's step weights.

**Cost.** R, the number of non-Clifford rotations, is the primary currency. For fault-tolerant accounting, each rotation is synthesised to Clifford+T gates. Assumption (not from the literature brief): precision epsilon costs c log2(1/epsilon) T gates, c = 3. Errors add in the spectral norm, so the total error is the algorithmic error plus R times the per-rotation synthesis error, giving T = R c log2(R / (target - algorithmic error)) when the algorithmic error is below the target and an unreachable budget otherwise. The cost of a method is the minimum T over its available depths. Because T is linear in c, changing c rescales all methods together.

## Methods compared

| Arm | Role | Description |
|---|---|---|
| Exact diagonalisation | classical ground truth | Dense eigendecomposition; defines the target and costs memory O(4^n) |
| Lanczos (Krylov) propagation | classical baseline | Propagates single states with m matrix-vector products; reference only, not cost-commensurate with a circuit |
| Strang and Suzuki product formulas | standard quantum baseline | Second- and fourth-order Trotter-Suzuki, K up to 40 diagonal layers (R up to 928 at n = 8) |
| Learned angles (proposed) | method | Strang skeleton with every angle free (n_params = R), K = 3-6, optimised from the Strang angles with Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) on loss 1 - |Tr(U^dag V)| / d, at most 80 iterations |

## Metrics

Main metric: spectral-norm error, the minimum over a global phase of ||V - e^{i phi} U||, computed exactly from the arc occupied by the eigenphases of U^dag V. Secondary: entanglement infidelity 1 - |Tr(U^dag V)|^2 / d^2, and the state error on random product states. Results are reported as mean +/- standard deviation over instances (5 seeds at n = 4 and n = 6 with t = 1, 3 elsewhere).

## Ablation (n = 6, t = 1.0, K = 5)

1. Free angles from the Strang start (the method).
2. Free angles from a random start (isolates initialisation).
3. Shared scales: angles = Strang angles times 3 learned scale factors per step (isolates per-site freedom; resembles optimised splitting coefficients).
4. Generic brickwork (RY, RZ, CZ layers) with at most the same rotation count, random start (isolates the product-formula skeleton).
5. Strang without learning.

## Noise and robustness sweep (n = 6, t = 1.0, K = 5, same circuits as the main result)

Arms: learned (K = 5); Strang with the same rotation count (K = 5); Strang with 2K steps (roughly equal algorithmic error, about twice the rotations); Suzuki fourth order with r = 2.

- **Coherent angle noise:** every rotation angle receives independent Gaussian noise of standard deviation sigma in {0, 0.002, 0.005, 0.01, 0.02, 0.05} rad (20 draws per instance); this stands in for control miscalibration or coarse synthesis.
- **Pauli noise:** each rotation is followed (at the end of its layer) by a uniformly random non-identity Pauli on its qubits with probability p in {0, 1e-3, 3e-3, 1e-2}; process infidelity averaged over 40 trajectories per instance. This is a layer-level approximation of per-gate depolarising noise.

## Seeds, compute and engineering

Instance seeds 0-4 (or 0-2); the learned-angle optimiser is deterministic given the start (the random-start ablation uses seed + 77). Torch uses 1 thread per worker and 4 worker processes. One command, `python experiments/run_all.py`, regenerates all of `results/` and `figures/`; `--quick` runs a smoke test in a temporary folder. Tests cover unitarity, convergence orders, metric invariances, cost-model monotonicity, the noise model and the Krylov baseline.
