import numpy as np
import pytest
from qdflow.physics import simulation

def test_calc_K_mat_shape_and_values():
    x = np.linspace(-10, 10, 5)
    K_0 = 2.0
    sigma = 1.0
    K_mat = simulation.calc_K_mat(x, K_0, sigma)
    assert K_mat.shape == (5, 5)
    # Diagonal should be K_0 / sigma
    assert np.allclose(np.diag(K_mat), K_0 / sigma)
    # Off-diagonal symmetry
    assert np.allclose(K_mat, K_mat.T)

def test_GateParameters_from_dict_and_copy():
    d = {"mean": 1.0, "peak": 2.0, "rho": 3.0, "h": 4.0, "screen": 5.0}
    gate = simulation.GateParameters.from_dict(d)
    assert gate.mean == 1.0
    assert gate.peak == 2.0
    assert gate.rho == 3.0
    assert gate.h == 4.0
    assert gate.screen == 5.0
    gate2 = gate.copy()
    assert gate2.mean == gate.mean

def test_PhysicsParameters_from_dict_and_copy():
    d = {
        "x": np.linspace(-1, 1, 3),
        "q": -1,
        "gates": [
            {"mean": 0, "peak": 1, "rho": 2, "h": 3, "screen": 4},
            {"mean": 1, "peak": 2, "rho": 3, "h": 4, "screen": 5},
        ],
        "K_0": 10.0,
        "sigma": 2.0,
    }
    phys = simulation.PhysicsParameters.from_dict(d)
    assert phys.x.shape == (3,)
    assert phys.q == -1
    assert isinstance(phys.gates[0], simulation.GateParameters)
    phys2 = phys.copy()
    assert np.allclose(phys2.x, phys.x)

def test_NumericsParameters_from_dict_and_copy():
    d = {"calc_n_max_iterations": 10, "calc_n_rel_tol": 1e-3}
    num = simulation.NumericsParameters.from_dict(d)
    assert num.calc_n_max_iterations == 10
    assert num.calc_n_rel_tol == 1e-3
    num2 = num.copy()
    assert num2.calc_n_max_iterations == num.calc_n_max_iterations

def test_ThomasFermiOutput_from_dict_and_copy():
    d = {
        "island_charges": np.array([1, 2]),
        "sensor": np.array([0.1, 0.2]),
        "are_dots_occupied": np.array([True, False]),
        "are_dots_combined": np.array([False]),
        "dot_charges": np.array([1, 0]),
        "converged": True,
        "n": np.array([0.0, 1.0]),
    }
    out = simulation.ThomasFermiOutput.from_dict(d)
    assert out.island_charges[0] == 1
    assert out.sensor[1] == 0.2
    out2 = out.copy()
    assert np.allclose(out2.n, out.n)

def test_calc_V_gate_scalar_and_array():
    gate = simulation.GateParameters(mean=0, peak=1, rho=2, h=3, screen=4)
    v_scalar = simulation.calc_V_gate(gate, 0, 0, 0)
    v_array = simulation.calc_V_gate(gate, np.array([0, 1]), 0, 0)
    assert isinstance(v_scalar, float)
    assert v_array.shape == (2,)

def test_calc_effective_peaks_shape():
    gates = [
        simulation.GateParameters(mean=0, peak=1, rho=2, h=3, screen=4),
        simulation.GateParameters(mean=1, peak=2, rho=3, h=4, screen=5),
    ]
    eff_peaks = simulation.calc_effective_peaks(gates)
    assert eff_peaks.shape == (2,)

def test_calc_V_shape():
    gates = [
        simulation.GateParameters(mean=0, peak=1, rho=2, h=3, screen=4),
        simulation.GateParameters(mean=1, peak=2, rho=3, h=4, screen=5),
    ]
    x = np.linspace(-1, 1, 5)
    v = simulation.calc_V(gates, x, 0, 0)
    assert v.shape == (5,)

def test_ThomasFermi_basic_run():
    phys = simulation.PhysicsParameters()
    tf = simulation.ThomasFermi(phys)
    n = tf.calc_n()
    assert n.shape == phys.x.shape
    assert isinstance(tf.converged, bool)

def test_is_transition_basic():
    dc1 = np.array([1, 0])
    adc1 = np.array([False])
    dc2 = np.array([2, 0])
    adc2 = np.array([False])
    is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
    assert is_tr.shape == (2,)
    assert is_tr_comb.shape == (1,)
    assert np.any(is_tr)

# Add more tests for edge cases and exceptions as needed
