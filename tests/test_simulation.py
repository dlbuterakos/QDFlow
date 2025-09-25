import numpy as np
import pytest
from qdflow.physics import simulation
import warnings

# ----------------
# Test Dataclasses
# ----------------

class TestDataclasses:

    @staticmethod
    def test_GateParameters():
        d = {"mean": 1.0, "peak": 2.0, "rho": 3.0, "h": 4.0, "screen": 5.0}
        gate = simulation.GateParameters.from_dict(d)
        for k, v in d.items():
            assert getattr(gate, k) == v        
        gate_copy = gate.copy()
        assert gate_copy.mean == gate.mean
        assert gate is not gate_copy
        to_d = gate.to_dict()
        for k, v in d.items():
            assert to_d[k] == v
        d2 = {"mean": 1.0, "peak": 2.0, "extraneous_key": 12345}
        gate2 = simulation.GateParameters.from_dict(d2)
        assert gate2.rho == simulation.GateParameters().rho
        assert not hasattr(gate2, "extraneous_key")

    @staticmethod
    def test_PhysicsParameters():
        d = {
            "x": np.linspace(-1, 1, 3),
            "V": None,
            "q": -1,
            "gates": [
                {"mean": 1, "peak": 2, "rho": 3, "h": 4, "screen": 5},
                {"mean": 6, "peak": 7, "rho": 8, "h": 9, "screen": 10},
            ],
            "K_0": 10.0,
            "sigma": 2.0,
            "extraneous_key": 12345
        }
        phys = simulation.PhysicsParameters.from_dict(d)
        assert phys.x.shape == (3,)
        assert np.all(phys.x == d["x"])
        assert phys.x is not d["x"]
        assert phys.q == -1
        assert phys.V == None
        assert isinstance(phys.gates[0], simulation.GateParameters)
        assert phys.g_0 == simulation.PhysicsParameters().g_0
        assert not hasattr(phys, "extraneous_key")
        
        # check for deep copy
        phys_copy = phys.copy()
        assert phys.q == phys_copy.q
        assert np.all(phys_copy.x == phys.x)
        assert phys_copy.x is not phys.x
        assert phys_copy is not phys
        assert phys_copy.gates[1].mean == phys.gates[1].mean
        assert phys_copy.gates[1] is not phys.gates[1]
        
        to_d = phys.to_dict()        
        assert np.all(to_d["x"] == d["x"])
        assert to_d["x"] is not d["x"]
        assert isinstance(to_d["gates"][0], dict)
        assert "peak" in to_d["gates"][0]
        
    @staticmethod
    def test_NumericsParameters():
        d = {
            "calc_n_max_iterations_no_guess": 1,
            "calc_n_max_iterations_guess": 2,
            "calc_n_rel_tol": .3,
            "calc_n_coulomb_steps": 4,
            "calc_n_use_combination_method": True,
            "island_relative_cutoff": .5,
            "island_min_occupancy": .6,
            "cap_model_matrix_softening": .7,
            "stable_config_N_limit": 8,
            "count_transitions_sigma": .9,
            "count_transitions_eps": .11,
            "create_graph_max_changes": 12
        }
        numer = simulation.NumericsParameters.from_dict(d)
        for k, v in d.items():
            assert getattr(numer, k) == v        
        numer_copy = numer.copy()
        assert numer_copy.calc_n_rel_tol == numer.calc_n_rel_tol
        assert numer is not numer_copy
        to_d = numer.to_dict()
        for k, v in d.items():
            assert to_d[k] == v
        d2 = {"island_relative_cutoff": 1.0, "extraneous_key": 12345}
        numer2 = simulation.NumericsParameters.from_dict(d2)
        assert numer2.calc_n_rel_tol == simulation.NumericsParameters().calc_n_rel_tol
        assert not hasattr(numer2, "extraneous_key")

    @staticmethod
    def test_ThomasFermiOutput():
        d = {
            "sensor": np.array([0.1, 0.2]),
            "are_dots_occupied": np.array([True, False]),
            "converged": True,
            "n": np.array([0.0, 1.0]),
            "graph_charge": None,
            "extraneous_key": 12345
        }

        out = simulation.ThomasFermiOutput.from_dict(d)
        assert np.all(out.n == d["n"])
        assert out.n is not d["n"]
        assert out.converged == True
        assert out.graph_charge == None
        assert out.are_dots_combined.shape == simulation.ThomasFermiOutput().are_dots_combined.shape
        assert np.all(out.are_dots_combined == simulation.ThomasFermiOutput().are_dots_combined)
        assert not hasattr(out, "extraneous_key")
        
        # check for deep copy
        out_copy = out.copy()
        assert out_copy is not out
        assert out.converged == out_copy.converged
        assert np.all(out_copy.n == out.n)
        assert out_copy.n is not out.n
        
        to_d = out.to_dict()
        assert np.all(to_d["n"] == d["n"])
        assert to_d["n"] is not d["n"]

# ---------------------------
# Test module-level functions
# ---------------------------

class TestModuleFunctions:

    @staticmethod
    def test_calc_K_mat():
        x = np.linspace(-5, 5, 7)
        K_0 = 2.0
        sigma = 1.0
        K_mat = simulation.calc_K_mat(x, K_0, sigma)
        assert K_mat.shape == (7, 7)
        assert np.allclose(np.diag(K_mat), K_0 / sigma)
        assert np.allclose(K_mat, K_mat.T)
        assert np.allclose(K_mat[:-1,:-1], K_mat[1:,1:])
        assert np.all(K_mat[0,:-1] > K_mat[0,1:])

    @staticmethod
    def test_calc_V_gate():
        gate1 = simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40)
        val1 = simulation.calc_V_gate(gate1, 0, -7, 7)
        val1b = simulation.calc_V_gate(gate1, 15, 0, 0)
        arr = simulation.calc_V_gate(gate1, np.array([0, 1, 2, 3]), -7, 7)
        assert isinstance(val1, float)
        assert np.isclose(val1b, 1.3)
        assert arr.shape == (4,)
        assert np.isclose(val1, arr[0])
        gate2 = simulation.GateParameters(mean=15, peak=2.6, rho=10, h=20, screen=40)
        val2 = simulation.calc_V_gate(gate2, 0, -7, 7)
        assert np.isclose(val2, 2 * val1)
        gate3 = simulation.GateParameters(mean=115, peak=1.3, rho=10, h=20, screen=40)
        val3 = simulation.calc_V_gate(gate3, 100, -7, 7)
        assert np.isclose(val3, val1)

    @staticmethod
    def test_calc_effective_peaks():
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40)
        ]
        eff_peaks = simulation.calc_effective_peaks(gates)
        assert eff_peaks.shape == (1,)
        assert np.isclose(eff_peaks[0], 1.3)
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40),
            simulation.GateParameters(mean=65, peak=1.3, rho=10, h=20, screen=40),            
        ]
        eff_peaks = simulation.calc_effective_peaks(gates)
        assert eff_peaks.shape == (2,)
        assert eff_peaks[0] < 1.3
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40),
            simulation.GateParameters(mean=65, peak=-1.3, rho=10, h=20, screen=40),            
        ]
        eff_peaks = simulation.calc_effective_peaks(gates)
        assert eff_peaks.shape == (2,)
        assert eff_peaks[0] > 1.3

    @staticmethod
    def test_calc_V():
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40),
            simulation.GateParameters(mean=65, peak=-1.3, rho=10, h=20, screen=40),            
        ]
        x = np.linspace(0, 100, 21)
        v = simulation.calc_V(gates, x, 0, 0)
        assert v.shape == (21,)
        assert v[13] < v[3]

    @staticmethod
    def test_is_transition():
        dc1 = np.array([1, 0])
        adc1 = np.array([False])
        dc2 = np.array([2, 0])
        adc2 = np.array([False])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (2,)
        assert is_tr_comb.shape == (1,)
        assert np.all(is_tr == [True, False])
        assert np.all(is_tr_comb == [False])

        dc1 = np.array([1, 2, 3])
        adc1 = np.array([False, False])
        dc2 = np.array([2, 1, 3])
        adc2 = np.array([False, False])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (3,)
        assert is_tr_comb.shape == (2,)
        assert np.all(is_tr == [True, True, False])
        assert np.all(is_tr_comb == [False, False])

        dc1 = np.array([2, 2])
        adc1 = np.array([False])
        dc2 = np.array([4, 0])
        adc2 = np.array([True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert np.all(is_tr == [False, False])
        assert np.all(is_tr_comb == [False])

        dc1 = np.array([2, 2])
        adc1 = np.array([False])
        dc2 = np.array([5, 0])
        adc2 = np.array([True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert np.all(is_tr == [True, True])
        assert np.all(is_tr_comb == [True])

        dc1 = np.array([4, 0, 2])
        adc1 = np.array([True, False])
        dc2 = np.array([2, 4, 0])
        adc2 = np.array([False, True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert np.all(is_tr == [False, False, False])
        assert np.all(is_tr_comb == [False, False])

        dc1 = np.array([4, 0, 1])
        adc1 = np.array([True, False])
        dc2 = np.array([2, 4, 0])
        adc2 = np.array([False, True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert np.all(is_tr == [True, True, True])
        assert np.all(is_tr_comb == [True, True])

        dc1 = np.array([4, 0, 2])
        adc1 = np.array([True, False])
        dc2 = np.array([5, 0, 1])
        adc2 = np.array([True, False])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert np.all(is_tr == [True, True, True])
        assert np.all(is_tr_comb == [True, False])

# ---------------------------
# Test ThomasFermi simulation
# ---------------------------

class TestThomasFermi:

    @staticmethod
    def test_calc_n():
        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        n = tf._calc_n()
        assert n.shape == (len(x),)
        assert np.all(n >= 0)

        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, K_0=0)
        V = simulation.calc_V(phys.gates, x, 0, 0)
        phys.V = V
        tf = simulation.ThomasFermi(phys)
        n = tf._calc_n()
        de = phys.mu - phys.q * phys.V
        n_exact = (phys.g_0 / phys.beta) * np.where(de > 0,
            phys.beta * de + np.log(1 + np.exp(phys.beta * np.where(de > 0, -de, 0))),
            np.log(1 + np.exp(phys.beta * np.where(de <= 0, de, 0)))
        )
        assert np.allclose(n, n_exact)

        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, K_0=10, g_0=.007, beta=1, mu=-20)
        V = 4 * phys.q * np.cos(2*np.pi*x/250)
        phys.V = V
        tf = simulation.ThomasFermi(phys)
        n = tf._calc_n()
        assert np.all(n >= 0)
        assert np.all(n < 1e-5)

    @staticmethod
    def test_calc_n_convergence_warning():
        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, g_0=.01, K_0=1000, sigma=20, beta=100)
        numer = simulation.NumericsParameters(calc_n_max_iterations_no_guess=50,
                                              calc_n_rel_tol=1e-4)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = tf._calc_n()
            assert len(w) == 1
            assert issubclass(w[0].category, simulation.ConvergenceWarning)
            assert tf.converged == False

        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, g_0=.001, K_0=1, sigma=20, beta=100)
        numer = simulation.NumericsParameters(calc_n_max_iterations_no_guess=200,
                                              calc_n_rel_tol=1e-4)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = tf._calc_n()
            assert len(w) == 0
            assert tf.converged == True

    @staticmethod
    def test_calc_islands_and_barriers():
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        numer = simulation.NumericsParameters(island_relative_cutoff=.1)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.n = np.array([.0050, .01,  .02,  .033, 2.4, 5.5, 6,   3.7,  2.8,  1.9,
                         .010,  .011, .012, 2.3,  4.4, 4.5, 2.6, .017, .008, .009])
        tf._calc_islands_and_barriers()
        isl = tf.islands
        bar = tf.barriers
        assert np.all(isl == [[4,10], [13,17]])
        assert np.all(bar == [[0,4], [10,13], [17,20]])

    @staticmethod
    def test_calc_charges():
        x = np.linspace(-300,300,20)
        delta_x = 600 / (20-1)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.01, .01, .01,  .01, 2, 3, 4,  5,  2,  2,
                         .01, .01, .01, 2,  4, 4, 2, .01, .01, .01])
        tf.islands = np.array([[4,10], [13,17]], dtype=np.int_)
        ch = tf._calc_charges()
        assert np.allclose(ch, delta_x * np.array([18, 12]))
        
    @staticmethod
    def test_calc_charge_centers():
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.01, .01, .01, .01, 20, 7, 1, 3, 3, 1,
                         .01, .01, .01, 2, 4, 4, 2, .01, .01, .01])
        tf.islands = np.array([[4,10], [13,17]], dtype=np.int_)
        tf._calc_charges()
        chc = tf._calc_charge_centers()
        assert np.allclose(chc, [x[5], (x[14]+x[15])/2])

    @staticmethod
    def test_calc_inv_cap_matrix():
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.01, .01, .01, .01, 20, 7, 1, 3, 3, 1,
                         .01, .01, .01, 2, 4, 4, 2, .01, .01, .01])
        tf.islands = np.array([[4,10], [13,17]], dtype=np.int_)
        tf._calc_charges()
        icm = tf._calc_inv_cap_matrix()
        assert icm.shape == (2,2)
        assert np.allclose(icm, icm.T)
        assert np.all(icm >= 0)

    @staticmethod
    def test_calc_cap_energy():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        tf.inv_cap_matrix = np.array([[20,10], [10,30]])
        tf.charges = np.array([2.4, 3.4])
        nvec1 = np.array([2,3])
        e1 = tf._calc_cap_energy(nvec1)
        nvec2 = np.array([2,4])
        e2 = tf._calc_cap_energy(nvec2)
        nvec3 = np.array([3,3])
        e3 = tf._calc_cap_energy(nvec3)
        assert np.isclose(e1, 11.2)
        assert np.isclose(e2, 9.2)
        assert np.isclose(e3, 7.2)

    @staticmethod
    def test_calc_stable_config():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        tf.inv_cap_matrix = np.array([[20,10], [10,30]])
        tf.charges = np.array([2.4, 3.4])
        sc = tf._calc_stable_config()
        assert np.all(sc == [3,3])

    @staticmethod
    def test_fermi():
        phys = simulation.PhysicsParameters(kT=.1)
        tf = simulation.ThomasFermi(phys)
        val = tf._fermi(0)
        arr = tf._fermi(np.array([-.1,0,.1]))
        assert isinstance(val, float)
        assert arr.shape == (3,)
        assert np.allclose(arr, [np.e/(1+np.e), .5, 1/(1+np.e)])

    @staticmethod
    def test_dfermi():
        phys = simulation.PhysicsParameters(kT=.1)
        tf = simulation.ThomasFermi(phys)
        val = tf._dfermi(0)
        e_arr = np.array([-.1,0,.1])
        arr = tf._dfermi(e_arr)
        a1 = tf._fermi(e_arr)
        eps = 1e-5
        a2 = tf._fermi(e_arr + eps)
        approx = (a2 - a1) / eps
        assert isinstance(val, float)
        assert arr.shape == (3,)
        assert np.allclose(arr, approx, rtol=.01)

