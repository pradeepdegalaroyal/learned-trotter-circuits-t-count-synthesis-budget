# Research gap: do learned simulation circuits still win when every rotation is paid for in T gates?

## State of the art (from the supplied literature)

**Simulation is the leading use case for fault-tolerant machines.** Daley et al. (2022, digital object identifier (DOI) 10.1038/s41586-022-04940-6) argue that practical quantum advantage will first appear in quantum simulation, analogue today and digital once fault-tolerant hardware exists. Digital simulation then rests on product formulas, whose rotation count grows as the target error shrinks.

**Fault tolerance changes what a gate costs.** Campbell, Terhal and Vuillot (2017, DOI 10.1038/nature23460) review how universal gate sets are built on error-corrected qubits and stress that the non-Clifford resources (magic states, colour-code tricks) dominate the overhead. Every arbitrary-angle rotation must be compiled into Clifford and T gates, so the number of rotations, and the precision demanded of each, is the cost that matters.

**Learning compact circuits is the newest alternative to product formulas.** Over et al. (2026, DOI 10.1088/2058-9565/aeb3b0) learn variational circuits for operators by back-propagation and report better resource scaling than Suzuki-Trotter expansions for propagators in quantum simulation and chemistry. Their cost is a circuit size in a continuous-angle gate set on idealised or hardware-connected circuits.

**Surrounding context.** Moll et al. (2018, DOI 10.1088/2058-9565/aab822) set the variational, error-mitigated near-term picture; Wang and Liu (2024, DOI 10.1088/1361-6633/ad7f69) map quantum machine learning from the noisy intermediate-scale quantum (NISQ) era to fault tolerance; Liu, Arunachalam and Temme (2021, DOI 10.1038/s41567-021-01287-z) show how rigorous speed-ups depend on fault-tolerant circuits; physical gate errors near 10^-3 or below are demonstrated by Ballance et al. (2016, DOI 10.1103/PhysRevLett.117.060504) and Gaebler et al. (2016, DOI 10.1103/PhysRevLett.117.060505); Wintersperger et al. (2023, DOI 10.1140/epjqt/s40507-023-00190-1) insist on end-to-end cost seen by the user. Recent algorithm papers on error-corrected machines (Lee, English and Bartlett, 2026, DOI 10.1038/s41534-026-01242-x on post-selection for quantum low-density parity-check codes) and on hybrid workflows (Duriez et al., 2026, DOI 10.1038/s41524-026-02059-0 on sample-based quantum diagonalisation; Shayegan, 2026, DOI 10.1038/s41598-026-40910-y on variational quantum linear solvers, the Harrow-Hassidim-Lloyd algorithm and annealing) quantify cost in logical error rates, abort rates or circuit depth.

## Related work

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Over et al., DOI 10.1088/2058-9565/aeb3b0 | 2026 | recent | Learns circuits for propagators; better scaling than Suzuki-Trotter | Cost in synthesised T gates; sensitivity to rotation errors; which part of the circuit (angles vs structure) carries the gain |
| Lee, English, Bartlett, DOI 10.1038/s41534-026-01242-x | 2026 | recent | Post-selection for quantum low-density parity-check codes, about 1000x lower logical error at 1% aborts | Algorithm-level cost of the logical circuit being protected |
| Duriez et al., DOI 10.1038/s41524-026-02059-0 | 2026 | recent | Sample-based quantum diagonalisation for band gaps on hardware | Fault-tolerant resource accounting |
| Shayegan, DOI 10.1038/s41598-026-40910-y | 2026 | recent | Compares variational, Harrow-Hassidim-Lloyd and annealing linear solvers by depth and noise | Learned circuits; T-count |
| Daley et al., DOI 10.1038/s41586-022-04940-6 | 2022 | landmark | Roadmap to practical advantage in quantum simulation | Concrete gate-level costs of compression methods |
| Campbell, Terhal, Vuillot, DOI 10.1038/nature23460 | 2017 | landmark | Universal gate constructions on error-corrected qubits; non-Clifford overhead | Algorithm-specific circuit compression |
| Moll et al., DOI 10.1088/2058-9565/aab822 | 2018 | landmark | Variational algorithms and simple error mitigation for near-term devices | Fault-tolerant cost model |
| Liu, Arunachalam, Temme, DOI 10.1038/s41567-021-01287-z | 2021 | landmark | Provable kernel speed-up robust to sampling errors | Learned operator compilation |
| Ballance et al., DOI 10.1103/PhysRevLett.117.060504 | 2016 | landmark | Two-qubit gate fidelity 99.9% in trapped ions | Algorithm design |
| Gaebler et al., DOI 10.1103/PhysRevLett.117.060505 | 2016 | landmark | Universal gate set, two-qubit error 8(4)x10^-4 | Algorithm design |
| Wang and Liu, DOI 10.1088/1361-6633/ad7f69 | 2024 | review | Quantum machine learning from noisy intermediate-scale to fault tolerance | Concrete T-count comparisons of learned versus analytic circuits |
| Wintersperger et al., DOI 10.1140/epjqt/s40507-023-00190-1 | 2023 | review | End-user view of neutral-atom hardware performance | Compilation of learned circuits |

## The gap

Product formulas are costed by rotation count and learned circuits (Over et al.) by gate count, but on an error-corrected machine the shared currency is the number of T gates after each rotation is synthesised to a precision that fits the total error budget (Campbell et al.). None of the supplied papers (i) puts learned and analytic circuits on this T-count scale, (ii) tests whether a learned circuit keeps its accuracy when its angles are perturbed or synthesised coarsely, or (iii) separates how much of the learned gain comes from freely tuned angles on a product-formula skeleton versus from a generic circuit structure.

**Novelty in one sentence:** this study measures, at equal non-Clifford rotation count and at equal synthesis-inclusive T-count, how much error a product-formula skeleton with freely learned angles removes relative to Strang and Suzuki formulas, and how that advantage responds to angle noise, Pauli noise and ablations of parameterisation and initialisation.

## Hypotheses and success metrics (simulation scale: 4-8 qubits, dense operators)

All thresholds below were fixed before the full runs; a single-instance pilot (instance seed 0, n = 6 and 8) was used only to choose the evolution times (a time of 2 was too long for the learned depths tried).

- **H1 (accuracy at equal rotation count).** For depths K = 5 and 6 (K is the number of diagonal layers) at n = 6, learned angles reach at least 2x lower mean spectral-norm error than the Strang formula with the same rotation count, for each evolution time t in {0.5, 1.0, 1.5}. Falsified if the ratio is below 2 for any t.
- **H2 (synthesis-inclusive cost).** At a total spectral-norm error target of 0.1, the cheapest learned circuit needs at least 1.5x fewer T gates (with 3 T gates per bit of rotation precision) than the cheapest product formula (Strang or Suzuki) at n = 6, t = 1.0. Falsified if the ratio is below 1.5 or if the target is not reached on every instance.
- **H3 (robustness).** At equal algorithmic error, the learned circuit's process infidelity under Pauli noise is no larger than that of the Strang circuit with more rotations, and under coherent angle noise its error stays within 2x of the equal-rotation-count Strang circuit. Falsified otherwise.
- **H4 (ablation).** Free per-site angles from a product-formula start are needed: a shared-scale parameterisation (3 numbers per step), a random start, and a generic brickwork circuit of at most the same rotation count all have at least 2x higher mean error.
