import numpy as np
import pytest
from qdflow.physics import noise
from qdflow.util import distribution


# ----------------
# Test Dataclasses
# ----------------

class TestDataclasses:
    
    @staticmethod
    def test_NoiseParameters():
        d = {
            "noise_axis": 1,
            "sech_blur_width": 1.5,
            "unint_dot_spacing": np.array([1,2,3]),
            "extraneous_key": 12345
        }
        params = noise.NoiseParameters.from_dict(d)
        assert params.unint_dot_spacing.shape == (3,)
        assert np.all(params.unint_dot_spacing == d["unint_dot_spacing"])
        assert params.unint_dot_spacing is not d["unint_dot_spacing"]
        assert np.isclose(params.sech_blur_width, 1.5)
        assert params.coulomb_peak_width is None
        assert not hasattr(params, "extraneous_key")
        
        # check for deep copy
        params_copy = params.copy()
        assert params.sech_blur_width == params_copy.sech_blur_width
        assert params_copy.unint_dot_spacing.shape == params.unint_dot_spacing.shape
        assert np.all(params_copy.unint_dot_spacing == params.unint_dot_spacing)
        assert params_copy.unint_dot_spacing is not params.unint_dot_spacing
        assert params_copy is not params
        
        to_d = params.to_dict()
        assert to_d["unint_dot_spacing"].shape == d["unint_dot_spacing"].shape
        assert np.all(to_d["unint_dot_spacing"] == d["unint_dot_spacing"])
        assert to_d["unint_dot_spacing"] is not d["unint_dot_spacing"]

    @staticmethod
    def test_NoiseRandomization():
        corr = distribution.FullyCorrelated(distribution.Uniform(1,5), 2).dependent_distributions()
        d = {
            "white_noise_magnitude":.3,
            "pink_noise_magnitude":distribution.Delta(.4),
            "telegraph_low_pixels":corr[0],
            "telegraph_high_pixels":corr[1],
            "extraneous_key": 12345
        }
        rand = noise.NoiseRandomization.from_dict(d)
        assert rand.white_noise_magnitude == .3
        assert rand.pink_noise_magnitude is d["pink_noise_magnitude"]
        assert rand.telegraph_low_pixels is corr[0]
        assert rand.sech_blur_width == noise.NoiseRandomization().sech_blur_width
        assert not hasattr(rand, "extraneous_key")
        
        # check deep copy
        rand_copy = rand.copy()
        assert rand_copy.white_noise_magnitude == rand.white_noise_magnitude
        assert rand_copy is not rand
        assert rand_copy.pink_noise_magnitude._value == rand.pink_noise_magnitude._value
        assert rand_copy.pink_noise_magnitude is not rand.pink_noise_magnitude
        assert rand_copy.telegraph_low_pixels is not rand.telegraph_low_pixels
        assert rand_copy.telegraph_high_pixels is not rand.telegraph_high_pixels
        assert rand_copy.telegraph_low_pixels.dependent_distributions[0] is rand_copy.telegraph_low_pixels
        assert rand_copy.telegraph_low_pixels.dependent_distributions[1] is rand_copy.telegraph_high_pixels
        assert rand_copy.telegraph_high_pixels.dependent_distributions[0] is rand_copy.telegraph_low_pixels

        to_d = rand.to_dict()
        assert to_d is not d
        assert to_d["white_noise_magnitude"] == d["white_noise_magnitude"]
        assert to_d["pink_noise_magnitude"]._value == d["pink_noise_magnitude"]._value
        assert to_d["pink_noise_magnitude"] is not d["pink_noise_magnitude"]
        assert to_d["telegraph_low_pixels"] is not rand.telegraph_low_pixels
        assert to_d["telegraph_high_pixels"] is not rand.telegraph_high_pixels
        assert to_d["telegraph_low_pixels"].dependent_distributions[0] is to_d["telegraph_low_pixels"]
        assert to_d["telegraph_low_pixels"].dependent_distributions[1] is to_d["telegraph_high_pixels"]
        assert to_d["telegraph_high_pixels"].dependent_distributions[0] is to_d["telegraph_low_pixels"]

        rand_def1 = noise.NoiseRandomization.default()
        rand_def2 = noise.NoiseRandomization.default()
        assert rand_def1 is not rand_def2
        assert rand_def1.n_gates == rand_def2.n_gates

# ------------------------
# Test random_noise_params
# ------------------------

def test_random_noise_params():
    noise.set_rng_seed(456)
    rand = noise.NoiseRandomization.default()
    rand.n_gates = 3
    rand.white_noise_magnitude = distribution.Uniform(.2,.3)
    rand.pink_noise_magnitude = .4
    rand.unint_dot_spacing = distribution.FullyCorrelated(distribution.Uniform(3,7), 3)
    rand.sensor_gate_coupling = distribution.Uniform(.1,.3)
    params = noise.random_noise_params(rand)
    assert isinstance(params, noise.NoiseParameters)
    assert params.white_noise_magnitude >= .2 and params.white_noise_magnitude <= .3
    assert np.isclose(params.pink_noise_magnitude, .4)
    assert params.unint_dot_spacing.shape == (3,)
    assert np.all((params.unint_dot_spacing >= 3) & (params.unint_dot_spacing <= 7))
    assert np.allclose(params.unint_dot_spacing, params.unint_dot_spacing[0])
    assert params.sensor_gate_coupling.shape == (3,)
    assert np.all((params.sensor_gate_coupling >= .1) & (params.sensor_gate_coupling <= .3))
    
    rand = noise.NoiseRandomization.default()
    rand.n_gates = 3
    rand.coulomb_peak_width = None
    rand.unint_dot_spacing = None
    rand.sensor_gate_coupling = np.array([.1,.2,.3])
    params = noise.random_noise_params(rand)
    assert isinstance(params, noise.NoiseParameters)
    assert params.coulomb_peak_width is None
    assert params.unint_dot_spacing is None
    assert params.sensor_gate_coupling.shape == (3,)
    assert np.allclose(params.sensor_gate_coupling, [.1,.2,.3])
    
# -------------------
# Test NoiseGenerator
# -------------------

# class TestNoiseGenerator:

#     @staticmethod
#     def test_white_noise():

# def make_simple_map(shape=(10, 10)):
#     return np.ones(shape)

# def test_white_noise():
#     ng = noise.NoiseGenerator(noise.NoiseParameters(white_noise_magnitude=0.1))
#     data = make_simple_map()
#     noisy = ng.white_noise(data, 0.1)
#     assert noisy.shape == data.shape
#     assert not np.allclose(noisy, data)

# def test_pink_noise():
#     ng = noise.NoiseGenerator(noise.NoiseParameters(pink_noise_magnitude=0.1))
#     data = make_simple_map()
#     noisy = ng.pink_noise(data, 0.1)
#     assert noisy.shape == data.shape
#     assert not np.allclose(noisy, data)

# def test_telegraph_noise():
#     ng = noise.NoiseGenerator(noise.NoiseParameters())
#     data = make_simple_map()
#     noisy = ng.telegraph_noise(data, 0.2, 0.1, 2, 2, axis=0)
#     assert noisy.shape == data.shape

# def test_line_shift():
#     ng = noise.NoiseGenerator(noise.NoiseParameters())
#     data = make_simple_map()
#     shifted = ng.line_shift(data, ave_pixels=2, axis=0, shift_positive=True)
#     assert shifted.shape == data.shape

# def test_sech_blur():
#     ng = noise.NoiseGenerator(noise.NoiseParameters())
#     data = make_simple_map()
#     blurred = ng.sech_blur(data, blur_width=2, noise_axis=0)
#     assert blurred.shape == data.shape

# def test_sensor_gate():
#     ng = noise.NoiseGenerator(noise.NoiseParameters())
#     data = make_simple_map()
#     sgc = np.array([0.1, 0.2])
#     noisy = ng.sensor_gate(data, sensor_gate_coupling=sgc)
#     assert noisy.shape == data.shape

# def test_unint_dot_add():
#     ng = noise.NoiseGenerator(noise.NoiseParameters())
#     data = make_simple_map()
#     spacing = np.array([1.0, 0.0])
#     noisy = ng.unint_dot_add(data, magnitude=0.1, spacing=spacing, width=1.0, offset=0.5)
#     assert noisy.shape == data.shape

# def test_high_coupling_coulomb_peak():
#     ng = noise.NoiseGenerator(noise.NoiseParameters())
#     data = make_simple_map()
#     out = ng.high_coupling_coulomb_peak(data, peak_offset=0.5, peak_width=1.0, peak_spacing=2.0)
#     assert out.shape == data.shape

# def test_calc_noisy_map_all_noise_types():
#     params = noise.NoiseParameters(
#         white_noise_magnitude=0.1,
#         pink_noise_magnitude=0.1,
#         telegraph_magnitude=0.1,
#         telegraph_stdev=0.05,
#         telegraph_low_pixels=2,
#         telegraph_high_pixels=2,
#         latching_pixels=1,
#         latching_positive=True,
#         sech_blur_width=1.0,
#         unint_dot_magnitude=0.1,
#         unint_dot_spacing=np.array([1.0, 0.0]),
#         unint_dot_width=1.0,
#         unint_dot_offset=0.5,
#         coulomb_peak_spacing=2.0,
#         coulomb_peak_offset=0.5,
#         coulomb_peak_width=1.0,
#         sensor_gate_coupling=np.array([0.1, 0.2]),
#         noise_axis=0
#     )
#     ng = noise.NoiseGenerator(params)
#     data = make_simple_map()
#     noisy = ng.calc_noisy_map(data)
#     assert noisy.shape == data.shape
