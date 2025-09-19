import numpy as np
import pytest
from qdflow.physics import noise

def make_simple_map(shape=(10, 10)):
    return np.ones(shape)

def test_white_noise():
    ng = noise.NoiseGenerator(noise.NoiseParameters(white_noise_magnitude=0.1))
    data = make_simple_map()
    noisy = ng.white_noise(data, 0.1)
    assert noisy.shape == data.shape
    assert not np.allclose(noisy, data)

def test_pink_noise():
    ng = noise.NoiseGenerator(noise.NoiseParameters(pink_noise_magnitude=0.1))
    data = make_simple_map()
    noisy = ng.pink_noise(data, 0.1)
    assert noisy.shape == data.shape
    assert not np.allclose(noisy, data)

def test_telegraph_noise():
    ng = noise.NoiseGenerator(noise.NoiseParameters())
    data = make_simple_map()
    noisy = ng.telegraph_noise(data, 0.2, 0.1, 2, 2, axis=0)
    assert noisy.shape == data.shape

def test_line_shift():
    ng = noise.NoiseGenerator(noise.NoiseParameters())
    data = make_simple_map()
    shifted = ng.line_shift(data, ave_pixels=2, axis=0, shift_positive=True)
    assert shifted.shape == data.shape

def test_sech_blur():
    ng = noise.NoiseGenerator(noise.NoiseParameters())
    data = make_simple_map()
    blurred = ng.sech_blur(data, blur_width=2, noise_axis=0)
    assert blurred.shape == data.shape

def test_sensor_gate():
    ng = noise.NoiseGenerator(noise.NoiseParameters())
    data = make_simple_map()
    sgc = np.array([0.1, 0.2])
    noisy = ng.sensor_gate(data, sensor_gate_coupling=sgc)
    assert noisy.shape == data.shape

def test_unint_dot_add():
    ng = noise.NoiseGenerator(noise.NoiseParameters())
    data = make_simple_map()
    spacing = np.array([1.0, 0.0])
    noisy = ng.unint_dot_add(data, magnitude=0.1, spacing=spacing, width=1.0, offset=0.5)
    assert noisy.shape == data.shape

def test_high_coupling_coulomb_peak():
    ng = noise.NoiseGenerator(noise.NoiseParameters())
    data = make_simple_map()
    out = ng.high_coupling_coulomb_peak(data, peak_offset=0.5, peak_width=1.0, peak_spacing=2.0)
    assert out.shape == data.shape

def test_calc_noisy_map_all_noise_types():
    params = noise.NoiseParameters(
        white_noise_magnitude=0.1,
        pink_noise_magnitude=0.1,
        telegraph_magnitude=0.1,
        telegraph_stdev=0.05,
        telegraph_low_pixels=2,
        telegraph_high_pixels=2,
        latching_pixels=1,
        latching_positive=True,
        sech_blur_width=1.0,
        unint_dot_mag=0.1,
        unint_dot_spacing=np.array([1.0, 0.0]),
        unint_dot_width=1.0,
        unint_dot_offset=0.5,
        coulomb_peak_spacing=2.0,
        coulomb_peak_offset=0.5,
        coulomb_peak_width=1.0,
        sensor_gate_coupling=np.array([0.1, 0.2]),
        noise_axis=0
    )
    ng = noise.NoiseGenerator(params)
    data = make_simple_map()
    noisy = ng.calc_noisy_map(data)
    assert noisy.shape == data.shape

def test_noise_parameters_from_dict_and_copy():
    d = {
        "white_noise_magnitude": 0.1,
        "pink_noise_magnitude": 0.2,
        "noise_axis": 1
    }
    params = noise.NoiseParameters.from_dict(d)
    assert params.white_noise_magnitude == 0.1
    assert params.pink_noise_magnitude == 0.2
    assert params.noise_axis == 1
    params2 = params.copy()
    assert params2.white_noise_magnitude == 0.1

def test_noise_randomization_default_and_copy():
    nr = noise.NoiseRandomization.default(q_positive=False)
    nr2 = nr.copy()
    assert nr.n_gates == nr2.n_gates
    assert isinstance(nr.white_noise_magnitude, object)

def test_random_noise_params():
    nr = noise.NoiseRandomization.default(q_positive=False)
    params = noise.random_noise_params(nr)
    assert isinstance(params, noise.NoiseParameters)
    assert hasattr(params, "white_noise_magnitude")
