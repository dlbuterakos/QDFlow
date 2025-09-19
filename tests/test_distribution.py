import numpy as np
import pytest
from qdflow.util import distribution

def test_normal_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Normal(5, 2)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=10)
    assert isinstance(val, float)
    assert arr.shape == (10,)

def test_uniform_draw():
    rng = np.random.default_rng(42)
    dist = distribution.Uniform(1, 3)
    val = dist.draw(rng)
    arr = dist.draw(rng, size=(2, 3))
    assert 1 <= val < 3
    assert arr.shape == (2, 3)
    assert np.all((arr >= 1) & (arr < 3))

def test_operator_overloading():
    rng = np.random.default_rng(42)
    dist1 = distribution.Normal(0, 1)
    dist2 = distribution.Uniform(1, 2)
    combined = dist1 + dist2
    val = combined.draw(rng)
    assert isinstance(val, float)

def test_fully_correlated():
    rng = np.random.default_rng(42)
    dist = distribution.Normal(0, 1)
    fc = distribution.FullyCorrelated(dist, 3)
    vals = fc.draw(rng)
    assert vals.shape == (3,)
    assert np.allclose(vals, vals[0])

def test_matrix_correlated():
    rng = np.random.default_rng(42)
    dists = [distribution.Normal(0, 1)]
    matrix = np.array([[1], [2]])
    mc = distribution.MatrixCorrelated(matrix, dists)
    vals = mc.draw(rng)
    assert vals.shape == (2,)
    assert np.isclose(vals[1], 2 * vals[0])

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

def test_abs_distribution():
    rng = np.random.default_rng(42)
    dist = distribution.Normal(-5, 1).abs()
    val = dist.draw(rng)
    assert val >= 0

def test_dependent_distribution_warning():
    rng = np.random.default_rng(42)
    dist = distribution.FullyCorrelated(distribution.Normal(0, 1), 2)
    d1, d2 = dist.dependent_distributions()
    v1 = d1.draw(rng, size=3)
    v2 = d2.draw(rng, size=3)
    assert np.allclose(v1, v2)

# Add more tests for edge cases and exceptions as needed
