import numpy as np
import pytest
from qdflow.util import distribution
import warnings

# ----------------------------------------
# Test draw() for individual distributions
# ----------------------------------------

def test_delta_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Delta(3)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=10)
    assert arr.shape == (10,)
    assert val == 3
    assert np.all(arr == 3)

def test_normal_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Normal(5, 2)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=(2,3,1))
    empty = dist.draw(rng, size=0)
    assert isinstance(val, float)
    assert arr.shape == (2,3,1)
    assert len(empty) == 0

def test_uniform_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Uniform(1, 3)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=(2, 3))
    assert 1 <= val < 3
    assert arr.shape == (2, 3)
    assert np.all((arr >= 1) & (arr < 3))

def test_binary_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Binary(0.5, "yes", "no")
    val = dist.draw(rng)
    arr = dist.draw(rng, size=10)
    assert val in ["yes", "no"]
    assert set(arr) <= {"yes", "no"}

def test_discrete_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Discrete(1, 4)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=10)
    assert 1 <= val < 4
    assert np.all((arr >= 1) & (arr < 4))

def test_loguniform_draw():
    rng = np.random.default_rng(42)
    dist = distribution.LogUniform(1, 3)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=20)
    assert 1 <= val <= 3
    assert np.all((arr >= 1) & (arr <= 3))

def test_lognormal_draw():
    rng = np.random.default_rng(42)
    dist = distribution.LogNormal(0, 10)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=20)
    assert isinstance(val, float)
    assert np.all(arr > 0)

# -------------------------
# Test operator overloading
# -------------------------

def test_add_overloading():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(3)
    dist2 = distribution.Delta(4)
    comb1 = dist1 + dist2
    comb2 = 7 + dist1
    comb3 = dist2 + 5
    val1 = comb1.draw(rng)
    val2 = comb2.draw(rng)
    val3 = comb3.draw(rng)
    assert val1 == 7
    assert val2 == 10
    assert val3 == 9
    dist3 = distribution.Uniform(13,19)
    dist4 = distribution.LogUniform(27,28)
    comb = dist3 + dist4
    arr = comb.draw(rng, size=20)
    assert np.all((arr >= 40) & (arr <= 47))

def test_sub_overloading():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(3)
    dist2 = distribution.Delta(4)
    comb1 = dist1 - dist2
    comb2 = 7 - dist1
    comb3 = dist2 - 6
    val1 = comb1.draw(rng)
    val2 = comb2.draw(rng)
    val3 = comb3.draw(rng)
    assert val1 == -1
    assert val2 == 4
    assert val3 == -2
    dist3 = distribution.Uniform(13,19)
    dist4 = distribution.LogUniform(27,28)
    comb = dist3 - dist4
    arr = comb.draw(rng, size=20)
    assert np.all((arr >= -15) & (arr <= -8))

def test_neg_overloading():
    rng = np.random.default_rng(42)
    dist = distribution.Delta(3)
    comb = -dist
    val = comb.draw(rng)
    assert val == -3
    dist2 = distribution.Uniform(13,19)
    comb = -dist2
    arr = comb.draw(rng, size=20)
    assert np.all((arr >= -19) & (arr <= -13))
    
def test_mul_overloading():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(3)
    dist2 = distribution.Delta(4)
    comb1 = dist1 * dist2
    comb2 = 7 * dist1
    comb3 = dist2 * 6
    val1 = comb1.draw(rng)
    val2 = comb2.draw(rng)
    val3 = comb3.draw(rng)
    assert val1 == 12
    assert val2 == 21
    assert val3 == 24
    dist3 = distribution.Uniform(-3,-2)
    dist4 = distribution.LogUniform(6,7)
    comb = dist3 * dist4
    arr = comb.draw(rng, size=20)
    assert np.all((arr >= -21) & (arr <= -12))

def test_div_overloading():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(18)
    dist2 = distribution.Delta(6)
    comb1 = dist1 / dist2
    comb2 = 27 / dist1
    comb3 = dist2 / 5
    val1 = comb1.draw(rng)
    val2 = comb2.draw(rng)
    val3 = comb3.draw(rng)
    assert np.isclose(val1, 3)
    assert np.isclose(val2, 1.5)
    assert np.isclose(val3, 1.2)
    dist3 = distribution.Uniform(12,20)
    dist4 = distribution.LogUniform(40,48)
    comb = dist3 / dist4
    arr = comb.draw(rng, size=20)
    assert np.all((arr >= .25) & (arr <= .5))

def test_distribution_abs():
    rng = np.random.default_rng(42)
    dist = distribution.Delta(-3)
    dist2 = distribution.Normal(-5, 1).abs()
    comb = dist.abs()
    val = comb.draw(rng)
    arr = dist2.draw(rng, 20)
    assert val == 3
    assert np.all(arr >= 0)

# --------------------------------
# Test operator overloading errors
# --------------------------------

def test_add_error():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(None)
    dist2 = distribution.Delta(None)
    bad = dist1 + dist2
    error = False
    try:
        val = bad.draw(rng)
    except ValueError:
        error = True
    assert error

def test_sub_error():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(None)
    dist2 = distribution.Delta(None)
    bad = dist1 - dist2
    error = False
    try:
        val = bad.draw(rng)
    except ValueError:
        error = True
    assert error

def test_mul_error():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(None)
    dist2 = distribution.Delta(None)
    bad = dist1 * dist2
    error = False
    try:
        val = bad.draw(rng)
    except ValueError:
        error = True
    assert error

def test_div_error():
    rng = np.random.default_rng(42)
    dist1 = distribution.Delta(None)
    dist2 = distribution.Delta(None)
    bad = dist1 / dist2
    error = False
    try:
        val = bad.draw(rng)
    except ValueError:
        error = True
    assert error

def test_neg_error():
    rng = np.random.default_rng(42)
    dist = distribution.Delta(None)
    bad = -dist
    error = False
    try:
        val = bad.draw(rng)
    except ValueError:
        error = True
    assert error

# -----------------------------
# Test correlated distributions
# -----------------------------

def test_fully_correlated():
    rng = np.random.default_rng(42)
    dist = distribution.Normal(9, 1)
    fc = distribution.FullyCorrelated(dist, 3)
    vals = fc.draw(rng)
    arr = fc.draw(rng, size=(2,4))
    assert vals.shape == (3,)
    assert np.allclose(vals, vals[0])
    assert arr.shape == (2,4,3)
    assert np.allclose(arr, np.expand_dims(arr[:,:,0],2))
    dep_dists = fc.dependent_distributions()
    assert len(dep_dists) == 3
    d1, d2, d3 = dep_dists
    v1 = d1.draw(rng)
    v2 = d2.draw(rng)
    v3 = d3.draw(rng)
    assert v1 == v2
    assert v1 == v3

def test_matrix_correlated():
    rng = np.random.default_rng(42)
    dists = [distribution.Normal(9, 1)]
    matrix = np.array([[1], [2]])
    mc = distribution.MatrixCorrelated(matrix, dists)
    vals = mc.draw(rng)
    arr = mc.draw(rng, size=11)
    assert vals.shape == (2,)
    assert np.isclose(vals[1], 2 * vals[0])
    assert arr.shape == (11,2)
    assert np.allclose(arr[:,1], 2 * arr[:,0])
    dists2 = [distribution.Normal(-27, 3), distribution.LogUniform(40,400)]
    mat2 = np.array([[1,0], [0,1], [-2,.5]])
    mc2 = distribution.MatrixCorrelated(mat2, dists2)
    dep_dists = mc2.dependent_distributions()
    assert len(dep_dists) == 3
    arrs = [d.draw(rng, size=20) for d in dep_dists]
    assert np.allclose(-2*arrs[0] + .5*arrs[1], arrs[2])

def test_spherically_correlated():
    rng = np.random.default_rng(42)
    sc = distribution.SphericallyCorrelated(5, radius=25)
    vals = sc.draw(rng)
    arr = sc.draw(rng, size=11)
    assert vals.shape == (5,)
    assert np.isclose(np.sqrt(np.sum(vals**2)), 25)
    assert arr.shape == (11,5)
    assert np.allclose(np.sqrt(np.sum(vals**2, axis=-1)), 25)
    dist = distribution.Uniform(9, 10)
    sc2 = distribution.SphericallyCorrelated(7, radius=dist)
    dep_dists = sc2.dependent_distributions()
    assert len(dep_dists) == 7
    arrs = [d.draw(rng, size=20) for d in dep_dists]
    rads = np.sqrt(np.sum(np.array(arrs)**2, axis=0))
    assert np.all((rads >= 9) | np.isclose(rads, 9))
    assert np.all((rads <= 10) | np.isclose(rads, 10))

# -----------------------------------
# Test correlated distribution errors
# -----------------------------------

def test_dependent_distribution_warning():
    rng = np.random.default_rng(42)
    dist = distribution.FullyCorrelated(distribution.Normal(9, 1), 2)
    d1, d2 = dist.dependent_distributions()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v1a = d1.draw(rng, size=3)
        v1b = d1.draw(rng, size=3)
        assert len(w) == 1
        assert issubclass(w[0].category, distribution.DependentDistributionWarning)
    dist = distribution.FullyCorrelated(distribution.Normal(13, 1), 2)
    d1, d2 = dist.dependent_distributions()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v1 = d1.draw(rng, size=3)
        v2 = d2.draw(rng, size=4)
        assert len(w) == 1
        assert issubclass(w[0].category, distribution.DependentDistributionWarning)
        
