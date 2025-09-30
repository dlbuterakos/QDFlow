import numpy as np
import pytest
from qdflow import generate
from qdflow.physics import simulation
from qdflow.util import distribution

# ----------------
# Test Dataclasses
# ----------------

class TestDataclasses:
    
    @staticmethod
    def test_CSDOutput():
        d = {
            "physics": {"mu":1.23, "sigma":20},
            "V_x": np.linspace(-1, 1, 3),
            "x_gate": 1,
            "excited_sensor": np.ones((3,4,2)),
            "dot_transitions": None,
            "extraneous_key": 12345
        }
        csd = generate.CSDOutput.from_dict(d)
        assert csd.V_x.shape == (3,)
        assert np.all(csd.V_x == d["V_x"])
        assert csd.V_x is not d["V_x"]
        assert csd.x_gate == 1
        assert csd.dot_transitions is None
        assert isinstance(csd.physics, simulation.PhysicsParameters)
        assert np.isclose(csd.physics.mu, 1.23)
        assert csd.converged is None
        assert not hasattr(csd, "extraneous_key")
        
        # check for deep copy
        csd_copy = csd.copy()
        assert csd.x_gate == csd_copy.x_gate
        assert csd_copy.V_x.shape == csd.V_x.shape
        assert np.all(csd_copy.V_x == csd.V_x)
        assert csd_copy.V_x is not csd.V_x
        assert csd_copy is not csd
        assert csd_copy.physics.mu == csd.physics.mu
        assert csd_copy.physics is not csd.physics
        
        to_d = csd.to_dict()
        assert to_d["V_x"].shape == d["V_x"].shape
        assert np.all(to_d["V_x"] == d["V_x"])
        assert to_d["V_x"] is not d["V_x"]
        assert isinstance(to_d["physics"], dict)
        assert "mu" in to_d["physics"]

    @staticmethod
    def test_RaysOutput():
        d = {
            "physics": {"mu":1.23, "sigma":20},
            "centers": np.array([[1,2],[3,4],[5,6]]),
            "resolution": 3,
            "excited_sensor": np.ones((3,2,3,1)),
            "dot_transitions": None,
            "extraneous_key": 12345
        }
        ro = generate.RaysOutput.from_dict(d)
        assert ro.centers.shape == (3,2)
        assert np.all(ro.centers == d["centers"])
        assert ro.centers is not d["centers"]
        assert ro.resolution == 3
        assert ro.dot_transitions is None
        assert isinstance(ro.physics, simulation.PhysicsParameters)
        assert np.isclose(ro.physics.mu, 1.23)
        assert ro.converged is None
        assert not hasattr(ro, "extraneous_key")
        
        # check for deep copy
        ro_copy = ro.copy()
        assert ro.resolution == ro_copy.resolution
        assert ro_copy.centers.shape == ro.centers.shape
        assert np.all(ro_copy.centers == ro.centers)
        assert ro_copy.centers is not ro.centers
        assert ro_copy is not ro
        assert ro_copy.physics.mu == ro.physics.mu
        assert ro_copy.physics is not ro.physics
        
        to_d = ro.to_dict()
        assert to_d["centers"].shape == d["centers"].shape
        assert np.all(to_d["centers"] == d["centers"])
        assert to_d["centers"] is not d["centers"]
        assert isinstance(to_d["physics"], dict)
        assert "mu" in to_d["physics"]
    
    @staticmethod
    def test_PhysicsRandomization():
        d = {
            "num_x_points":20,
            "mu":1.23,
            "sigma":distribution.Delta(20),
            "extraneous_key": 12345
        }
        rand = generate.PhysicsRandomization.from_dict(d)
        assert rand.mu == 1.23
        assert rand.num_x_points == 20
        assert rand.sigma is d["sigma"]
        assert rand.c_k == generate.PhysicsRandomization().c_k
        assert not hasattr(rand, "extraneous_key")
        
        # not deep copy (because of distributions)
        rand_copy = rand.copy()
        assert rand_copy.mu == rand.mu
        assert rand_copy is not rand
        assert rand_copy.sigma is rand.sigma
        
        # is a deep copy, as_dict() uses deep_copy()
        to_d = rand.to_dict()
        assert to_d["mu"] == d["mu"]
        assert to_d["sigma"] is not d["sigma"]


