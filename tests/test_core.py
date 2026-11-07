import math

import numpy as np
import pytest
import torch

from learned_trotter.circuits import build_unitary, hea_spec, rotation_count, strang_spec
from learned_trotter.ftcost import t_count
from learned_trotter.hamiltonian import dense_hamiltonian, exact_propagator, make_instance
from learned_trotter.krylov import lanczos_propagate
from learned_trotter.learn import fit
from learned_trotter.metrics import spectral_error, trace_infidelity
from learned_trotter.product_formulas import strang, suzuki4

N, T = 4, 1.5


def _err(inst, spec, ang):
    u = exact_propagator(inst, T)
    return spectral_error(u, build_unitary(ang, spec, inst.n).numpy())


def test_unitary_is_unitary():
    inst = make_instance(N, 0)
    spec, ang = strang(inst, T, 3)
    v = build_unitary(ang, spec, N).numpy()
    assert np.allclose(v.conj().T @ v, np.eye(2**N), atol=1e-10)
    hv = build_unitary(torch.randn(80, dtype=torch.float64) * 0.3, hea_spec(2), N).numpy()
    assert np.allclose(hv.conj().T @ hv, np.eye(2**N), atol=1e-10)


def test_strang_converges_second_order():
    inst = make_instance(N, 1)
    e = [_err(inst, *strang(inst, T, k)) for k in (4, 8, 16)]
    assert e[0] > e[1] > e[2]
    assert 3.0 < e[0] / e[1] < 5.0  # ~4x per halving of the step


def test_suzuki4_beats_strang_at_equal_rotation_count_for_small_steps():
    inst = make_instance(N, 2)
    s_spec, s_ang = strang(inst, T, 20)
    z_spec, z_ang = suzuki4(inst, T, 4)
    assert rotation_count(s_spec, N) == rotation_count(z_spec, N)
    assert _err(inst, z_spec, z_ang) < _err(inst, s_spec, s_ang)


def test_rotation_count_formula():
    n, k = 6, 3
    assert rotation_count(strang_spec(k), n) == k * (3 * n - 1) + n


def test_metrics_phase_invariant_and_zero():
    inst = make_instance(N, 0)
    u = exact_propagator(inst, T)
    assert spectral_error(u, u) < 1e-7
    assert spectral_error(u, np.exp(0.7j) * u) < 1e-7
    assert trace_infidelity(u, np.exp(0.7j) * u) < 1e-12
    assert spectral_error(u, np.eye(2**N)) > 0.1


def test_exact_propagator_matches_hamiltonian_derivative():
    inst = make_instance(3, 0)
    h = dense_hamiltonian(inst)
    dt = 1e-6
    u = exact_propagator(inst, dt)
    assert np.allclose((u - np.eye(8)) / dt, -1j * h, atol=1e-4)


def test_learning_never_worse_than_warm_start_and_improves():
    inst = make_instance(N, 3)
    u = exact_propagator(inst, T)
    spec, ang = strang(inst, T, 2)
    e0 = _err(inst, spec, ang)
    th, loss, _ = fit(spec, N, u, ang.numpy(), max_iter=60)
    e1 = spectral_error(u, build_unitary(torch.tensor(th), spec, N).numpy())
    assert e1 < e0


def test_t_count_monotone():
    assert t_count(50, 0.1, 0.05) == math.inf
    assert t_count(50, 0.01, 0.05) < t_count(50, 0.03, 0.05) < t_count(60, 0.03, 0.05) * 2
    assert t_count(50, 0.0, 0.05, c=6) == pytest.approx(2 * t_count(50, 0.0, 0.05, c=3))


def test_pauli_noise_zero_p_is_ideal_and_p_one_degrades():
    inst = make_instance(N, 0)
    spec, ang = strang(inst, T, 2)
    rng = np.random.default_rng(0)
    ideal = build_unitary(ang, spec, N).numpy()
    same = build_unitary(ang, spec, N, pauli_p=0.0, rng=rng).numpy()
    noisy = build_unitary(ang, spec, N, pauli_p=1.0, rng=rng).numpy()
    assert np.allclose(ideal, same)
    assert spectral_error(ideal, noisy) > 0.1


def test_lanczos_converges():
    inst = make_instance(N, 0)
    h = dense_hamiltonian(inst)
    u = exact_propagator(inst, T)
    psi = np.random.default_rng(0).standard_normal(2**N) + 0j
    psi /= np.linalg.norm(psi)
    e5 = np.linalg.norm(lanczos_propagate(h, psi, T, 5) - u @ psi)
    e16 = np.linalg.norm(lanczos_propagate(h, psi, T, 16) - u @ psi)
    assert e16 < 1e-8 < e5
