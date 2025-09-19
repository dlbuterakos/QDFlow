import numpy as np
from typing import Any, Self, overload, ClassVar, TypeVar
from numpy.typing import NDArray
from physics import simulation as simulation
import util.distribution as distribution
import dataclasses
from dataclasses import dataclass, field

T = TypeVar('T')


_rng = np.random.default_rng()

def set_rng_seed(seed):
    '''
    Initializes a new random number generator with the given seed,
    used to generate random data.

    Parameters
    ----------
    seed : {int, array_like[int], SeedSequence, BitGenerator, Generator}
        The seed to use to initialize the random number generator.
    '''
    global _rng
    _rng = np.random.default_rng(seed)


@dataclass(kw_only=True)
class CSDOutput:
    '''
    Output of charge stability diagram calculations. Some attributes may be ``None``
    depending on which quantities are calculated.

    Output values are stored in arrays such that the output for a property at point
    ``(V_x_vec[i], V_y_vec[j])`` is given by ``property[i][j]``.

    Attributes
    ----------
    
    physics : PhysicsParameters
        The set of physics parameters used in the simulation. 
    V_x_vec, V_y_vec : ndarray[float]
        Arrays of voltage values along the x- and y-axes.
    x_gate, y_gate : int
        The dot numbers of the dots whose gate voltages are plotted on the x- or y-axes.
    V_gates : ndarray[float]
        An array of length `n_dots` giving the voltages of each of the
        plunger gates (not barrier gates).
    sensor : ndarray[float]
        The Coulomb potential at each sensor.
    are_dots_occupied : ndarray[bool]
        An array of booleans, one for each dot, indicating whether each dot is occupied.    
    are_dots_combined : ndarray[bool]
        An array of booleans, one for each internal barrier,
        indicating whether the dots on each side are combined together
        (i.e. the barrier is too low).
    dot_charges : ndarray[int]
        An array of integers, one for each dot, indicating the total number
        of charges in each dot. In the case of combined dots, the
        total number of charges will be entered in the left-most dot,
        with the other dots padded with zeros.
    converged : ndarray[bool]
        Whether the calculation of n(x) properly converged.
    dot_transitions : ndarray[bool]
        Whether a transition is present at each dot.
    are_transitions_combined : ndarray[bool]
        Whether a combined transition is present on either side of each barrier.
    '''
    physics:simulation.PhysicsParameters=field(default_factory=lambda:simulation.PhysicsParameters())
    V_x_vec:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    V_y_vec:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    x_gate:int=0
    y_gate:int=0
    V_gates:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    sensor:NDArray[np.float32]=field(default_factory=lambda:np.zeros(0, dtype=np.float32))
    are_dots_occupied:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    are_dots_combined:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    dot_charges:NDArray[np.int_]=field(default_factory=lambda:np.zeros(0, dtype=np.int_))
    converged:NDArray[np.bool_]|None=None
    dot_transitions:NDArray[np.bool_]|None=None
    are_transitions_combined:NDArray[np.bool_]|None=None
    excited_sensor:NDArray[np.float32]|None=None
    
    def _get_physics(self) -> simulation.PhysicsParameters:
        return self._physics
    def _set_physics(self, val:simulation.PhysicsParameters):
        self._physics = val.copy()

    def _get_V_x_vec(self) -> NDArray[np.float64]:
        return self._V_x_vec
    def _set_V_x_vec(self, val:NDArray[np.float64]):
        self._V_x_vec = np.array(val, dtype=np.float64)

    def _get_V_y_vec(self) -> NDArray[np.float64]:
        return self._V_y_vec
    def _set_V_y_vec(self, val:NDArray[np.float64]):
        self._V_y_vec = np.array(val, dtype=np.float64)

    def _get_V_gates(self) -> NDArray[np.float64]:
        return self._V_gates
    def _set_V_gates(self, val:NDArray[np.float64]):
        self._V_gates = np.array(val, dtype=np.float64)

    def _get_sensor(self) -> NDArray[np.float32]:
        return self._sensor
    def _set_sensor(self, val:NDArray[np.float32]):
        self._sensor = np.array(val, dtype=np.float32)

    def _get_converged(self) -> NDArray[np.bool_]|None:
        return self._converged
    def _set_converged(self, val:NDArray[np.bool_]|None):
        self._converged = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_dots_occupied(self) -> NDArray[np.bool_]:
        return self._are_dots_occupied
    def _set_are_dots_occupied(self, val:NDArray[np.bool_]):
        self._are_dots_occupied = np.array(val, dtype=np.bool_)

    def _get_are_dots_combined(self) -> NDArray[np.bool_]:
        return self._are_dots_combined
    def _set_are_dots_combined(self, val:NDArray[np.bool_]):
        self._are_dots_combined = np.array(val, dtype=np.bool_)

    def _get_dot_charges(self) -> NDArray[np.int_]:
        return self._dot_states
    def _set_dot_charges(self, val:NDArray[np.int_]):
        self._dot_states = np.array(val, dtype=np.int_)

    def _get_dot_transitions(self) -> NDArray[np.bool_]|None:
        return self._dot_transitions
    def _set_dot_transitions(self, val:NDArray[np.bool_]|None):
        self._dot_transitions = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_transitions_combined(self) -> NDArray[np.bool_]|None:
        return self._are_transitions_combined
    def _set_are_transitions_combined(self, val:NDArray[np.bool_]|None):
        self._are_transitions_combined = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_excited_sensor(self) -> NDArray[np.float32]|None:
        return self._excited_sensor
    def _set_excited_sensor(self, val:NDArray[np.float32]|None):
        self._excited_sensor = np.array(val, dtype=np.float32) if val is not None else None


    @classmethod
    def from_dict(cls, d:dict[str, Any]) -> Self:
        '''
        Creates a new ``CSDOutput`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        CSDOutput
            A new ``CSDOutput`` object with the values specified by ``dict``.
        '''
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output
    

    def to_dict(self) -> dict[str, Any]:
        '''
        Converts the ``CSDOutput`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``CSDOutput`` object.
        '''
        return dataclasses.asdict(self)
    

    def copy(self) -> Self:
        '''
        Creates a copy of a ``CSDOutput`` object.

        Returns
        -------
        CSDOutput
            A new ``CSDOutput`` object with the same attribute values as ``self``.
        '''
        return dataclasses.replace(self)

CSDOutput.physics = property(CSDOutput._get_physics, CSDOutput._set_physics) # type: ignore
CSDOutput.V_x_vec = property(CSDOutput._get_V_x_vec, CSDOutput._set_V_x_vec) # type: ignore
CSDOutput.V_y_vec = property(CSDOutput._get_V_y_vec, CSDOutput._set_V_y_vec) # type: ignore
CSDOutput.V_gates = property(CSDOutput._get_V_gates, CSDOutput._set_V_gates) # type: ignore
CSDOutput.sensor = property(CSDOutput._get_sensor, CSDOutput._set_sensor) # type: ignore
CSDOutput.converged = property(CSDOutput._get_converged, CSDOutput._set_converged) # type: ignore
CSDOutput.are_dots_occupied = property(CSDOutput._get_are_dots_occupied, CSDOutput._set_are_dots_occupied) # type: ignore
CSDOutput.are_dots_combined = property(CSDOutput._get_are_dots_combined, CSDOutput._set_are_dots_combined) # type: ignore
CSDOutput.dot_charges = property(CSDOutput._get_dot_charges, CSDOutput._set_dot_charges) # type: ignore
CSDOutput.dot_transitions = property(CSDOutput._get_dot_transitions, CSDOutput._set_dot_transitions) # type: ignore
CSDOutput.are_transitions_combined = property(CSDOutput._get_are_transitions_combined, CSDOutput._set_are_transitions_combined) # type: ignore
CSDOutput.excited_sensor = property(CSDOutput._get_excited_sensor, CSDOutput._set_excited_sensor) # type: ignore



@dataclass(kw_only=True)
class PhysicsRandomization:
    '''
    Meta-parameters used to determine how random ``PhysicsParameters`` should
    be generated.

    Several attributes will not be randomized, and will be passed directly
    to the generated ``PhysicsParameters`` object.

    All other attributes should either be provided a single value
    (if no randmization is needed), or a ``distribution.Distribution`` object,
    from which the value will be drawn.
    
    Attributes
    ----------
    num_x_points : int
        The resolution of the x-axis. This value is not randomized.
    num_dots : int
        The number of dots. This value is not randomized.
    barrier_current : float
        An arbitrary low current set to the device when in barrier mode.
        This value is not randomized.
    short_circuit_current : float
        An arbitrary high current value given to the device when in
        open / short circuit mode. This value is not randomized.
    num_sensors : int
        The number of sensors to include. This value is not randomized.
    multiply_gates_by_q : bool
        Whether to multiply `barrier_peak`, `plunger_peak`, `barrier_peak_variations`,
        `plunger_peak_variations`, and `external_barrier_peak_variations` by `q`,
        changing the sign if ``q == -1``. Default True.
    dot_spacing : float | Distribution[float]
        The average distance (in nm) between dots.
    x_margins : float | Distribution[float]
        The length (in nm) of the nanowire to model on either end of the system.
        The total length of the nanowire will be:
        ``2 * (x_margins) + (num_dots - 1) * (dot_spacing)``.
    gate_x_rel_variations : float | Distribution[float]
        Each individual gate will draw a value from this distribution, multiply
        by ``dot_spacing``, and add to the x-coordinate of its center.
        This allows gates to be placed slightly off-center.
        Values drawn from this distribution should be significantly smaller than 1.
    q : float | Distribution[float]
        The charge of a particle, -1 for electrons, +1 for holes.
    K_0 : float | Distribution[float]
        The electron-electron Coulomb interaction strength (in meV * nm)
    sigma : float | Distribution[float]
        The softening parameter (in nm) for the el-el Coulomb interaction used
        to avoid divergence when x = x'. `sigma` should be on the scale of
        the width of the nanowire.
    mu : float | Distribution[float]
        The Fermi level (in meV)
    g_0 : float | Distribution[float]
        The coefficient of the density of states
    V_L : float | Distribution[float]
        The voltage applied to left lead (in mV).
    V_R : float | Distribution[float]
        The voltage applied to right lead (in mV).
    beta : float | Distribution[float]
        The inverse temperature ``1/(k_B T)`` used to calculate ``n(x)``.
    kT : float | Distribution[float]
        The temperature ``(k_B T)`` used in the transport calculations.
    c_k : float | Distribution[float]
        The coefficient (in meV*nm) that determines the kenetic energy of the
        Fermi sea on each island.
    screening_length : float | Distribution[float]
        The screening length (in nm) for the Coulomb interaction.
    rho : float | Distribution[float]
        The radius (in nm) of the cylindrical gates.
    h : float | Distribution[float]
        The distance (in nm) of the gates from the nanowire.
    rho_rel_variations : float | Distribution[float]
        Each individual gate will draw a value from this distribution, multiply
        by ``rho``, and add the result to ``rho``.
        This allows gates to havea slightly different sizes.
        Values drawn from this distribution should be significantly smaller than 1.
    h_variations : float | Distribution[float]
        Each individual gate will draw a value from this distribution, multiply
        by ``h``, and add the result to ``h``.
        This allows gates to have slightly different sizes.
        Values drawn from this distribution should be significantly smaller than 1.
    plunger_peak : float | Distribution[float]
        The peak value (in mV) of the potential at the nanowire due to the
        plunger gates. Each plunger will draw a different value.
    barrier_peak : float | Distribution[float]
        The peak value (in mV) of the potential at the nanowire due to the
        barrier gates.
    external_barrier_peak_variations : float | Distribution[float]
        Each external barrier gate will draw a different value from this
        distribution, and add the result to ``barrier_peak``.
        This allows barrier gates to have slightly different peak values.
    barrier_peak_variations : float | Distribution[float]
        Each internal barrier gate will draw a different value from this
        distribution, and add the result to ``barrier_peak``.
        This allows barrier gates to have slightly different peak values.
    sensor_y : float | Distribution[float]
        The y-coordinate (in nm) of the sensors.
    sensor_y_rel_variation : float | Distribution[float]
        Each individual sensor will draw a value from this distribution, multiply
        by ``sensor_y``, and add the result to its y-coordinate.
        This allows sensors to be placed at different distances.
        Values drawn from this distribution should be significantly smaller than 1.
    sensor_x_rel_variation : float | Distribution[float]
        Each individual sensor will draw a value from this distribution, multiply
        by half the distance between sensors, and add the result to its x-coordinate.
        This allows sensors to be slightly unevenly spaced.
        Values drawn from this distribution should be less than 1.
    WKB_coef : float | Distribution[float]
        Coefficient (with units 1/(nm*sqrt(meV))) which goes in the exponent
        while calculating the WKB probability, setting the strength of WKB tunneling.
        WKB_coef should be equal to ``sqrt(2*m)/hbar``
        (converted to units of 1/(nm*sqrt(meV))), where ``m`` is the effective
        mass of a particle, and ``hbar`` is the reduced Planck's constant.
    v_F : float | Distribution[float]
        The fermi velocity (in nm/s).
    '''
    num_x_points:int=151
    num_dots:int=2
    barrier_current:float=1e-5
    short_circuit_current:float=1e4
    num_sensors:int=1
    multiply_gates_by_q:bool=True

    dot_spacing:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(200))
    x_margins:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(200))
    gate_x_rel_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(0))
    q:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(-1))
    K_0:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.LogUniform(.5, 60))
    sigma:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(40, 80))
    mu:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('mu'))
    g_0:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.LogUniform(.005, .009))
    V_L:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('V_L'))
    V_R:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('V_R'))
    beta:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('beta'))
    kT:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('kT'))
    c_k:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.LogUniform(.25, 6))
    screening_length:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('scr'))
    rho:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(10, 20))
    h:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('h'))
    rho_rel_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(0))
    h_rel_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Normal(0,.15))
    plunger_peak:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(-12, -2))
    external_barrier_peak_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(.5, 3.5))
    barrier_peak:float|distribution.Distribution[float]=field(default_factory=lambda:PhysicsRandomization._correlated_default('bar'))
    barrier_peak_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(-1.5, 1.5))
    sensor_y:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(-250))
    sensor_y_rel_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(-.5, .5))
    sensor_x_rel_variations:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Uniform(-.35, .35))
    WKB_coef:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(.089))
    v_F:float|distribution.Distribution[float]=field(default_factory=lambda:distribution.Delta(3.0e13))
    
    _correlated_defaults:ClassVar[dict[str,distribution.Distribution[float]]]


    def __post_init__(self):
        PhysicsRandomization._prepare_correlated_defaults()


    @classmethod
    def _correlated_default(cls, key:str) -> distribution.Distribution[float]:
        return cls._correlated_defaults[key]


    @classmethod
    def _prepare_correlated_defaults(cls):
        cls._correlated_defaults = {}
        V_L = distribution.FullyCorrelated(distribution.Uniform(-.02, .02), 2).dependent_distributions()
        cls._correlated_defaults['V_L'] = V_L[0]
        cls._correlated_defaults['V_R'] = -V_L[1]
        beta = distribution.FullyCorrelated(distribution.LogUniform(10, 1000), 2).dependent_distributions()
        cls._correlated_defaults['beta'] = beta[0]
        cls._correlated_defaults['kT'] = 1/beta[1]
        h_scr_bar = distribution.MatrixCorrelated(np.array([[1,0],[0,1],[.09,.0125],[.05,0]]), [
                    distribution.Normal(80, 15).abs(), distribution.LogUniform(60, 200)]).dependent_distributions()
        cls._correlated_defaults['h'] = h_scr_bar[0]
        cls._correlated_defaults['scr'] = h_scr_bar[1]
        cls._correlated_defaults['bar'] = h_scr_bar[2]-3.5
        cls._correlated_defaults['mu'] = h_scr_bar[3]-3.5


    @classmethod
    def default(cls) -> Self:
        '''
        Creates a new ``PhysicsRandomizationParameters`` object with default values.

        Returns
        -------
        PhysicsRandomizationParameters
            A new ``PhysicsRandomizationParameters`` object with default values.
        '''
        return cls()


    @classmethod
    def from_dict(cls, d:dict[str, Any]) -> Self:
        '''
        Creates a new ``PhysicsRandomizationParameters`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        PhysicsRandomizationParameters
            A new ``PhysicsRandomizationParameters`` object with the values specified by ``dict``.
        '''
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output
    

    def to_dict(self) -> dict[str, Any]:
        '''
        Converts the ``PhysicsRandomizationParameters`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``PhysicsRandomizationParameters`` object.
        '''
        return dataclasses.asdict(self)
    

    def copy(self) -> Self:
        '''
        Creates a copy of a ``PhysicsRandomizationParameters`` object.

        Returns
        -------
        CSDOutput
            A new ``PhysicsRandomizationParameters`` object with the same attribute values as ``self``.
        '''
        return dataclasses.replace(self)

PhysicsRandomization._prepare_correlated_defaults()



def default_physics(n_dots:int=2) -> simulation.PhysicsParameters:
    '''
    Creates a new ``PhysicsParameters`` object initialized to a set of default values.

    Parameters
    ----------
    n_dots : int
        The number of dots in the device to model.

    Returns
    -------
    simulation.PhysicsParameters
        A default set of physics parameters.
    '''
    dot_points = 40
    edge_points = 40
    delta_x = 5.
    N_grid = 2*edge_points + (n_dots-1)*dot_points + 1
    system_size = (N_grid-1)*delta_x
    x = np.linspace(-system_size/2, system_size/2, N_grid, endpoint=True)

    physics = simulation.PhysicsParameters(
        x=x, q=-1, K_0=2, sigma=60, g_0=.01, beta=100, kT=.01, c_k=2,
        V_L=0, V_R=0, screening_length=100, mu=0,
        WKB_coef=.02, v_F=3.0e13, barrier_current=1e-5,
        short_circuit_current=1e4
    )
    
    def gate_peak(i):
        if i == 0 or i == 2*n_dots:
            return -8
        elif i % 2 == 0:
            return -5
        else:
            return 5
    gates = [simulation.GateParameters(mean=(i-n_dots)*dot_points*delta_x/2, 
                    peak=gate_peak(i), rho=20, h=60, screen=100)
             for i in range(2*n_dots+1)]
    physics.gates = gates
    physics.sensors=np.array([[(i-n_dots)*dot_points*delta_x/2, 200, 0] for i in range(2*n_dots+1)])
    return physics



@overload
def random_physics(randomization_params:PhysicsRandomization, num_physics:int) -> list[simulation.PhysicsParameters]: ...
@overload
def random_physics(randomization_params:PhysicsRandomization, num_physics:None=...) -> simulation.PhysicsParameters: ...

def random_physics(randomization_params:PhysicsRandomization, num_physics:int|None=None) \
                  -> simulation.PhysicsParameters|list[simulation.PhysicsParameters]:
    '''
    Creates a randomized set of physics parameters describing a QD device.

    Parameters
    ----------
    randomization_params : PhysicsRandomizationParameters
        Meta-parameters which indicate how the ``PhysicsParameters`` should be
        randomized.
    num_physics : int
        The number of ``PhysicsParameters`` sets to generate. 

    Returns
    -------
    simulation.PhysicsParameters
        The randomized set of physics parameters.
    '''
    global _rng
    r_p = randomization_params
    n_phys = 1 if num_physics is None else num_physics
    output = []

    @overload
    def draw(dist:T|distribution.Distribution[T], rng:np.random.Generator, size:None=...) -> T: ...
    @overload
    def draw(dist:T|distribution.Distribution[T], rng:np.random.Generator, size:int|tuple[int, ...]) -> NDArray: ...
    def draw(dist:T|distribution.Distribution[T], rng:np.random.Generator, size:int|tuple[int, ...]|None=None) -> T|NDArray:
        if isinstance(dist, distribution.Distribution):
            return dist.draw(rng, size)
        else:
            if size is None:
                return dist
            else:
                return np.full(size, dist) 

    for phys_i in range(n_phys):
        n_dots = r_p.num_dots
        physics = default_physics(n_dots)
        physics.barrier_current = r_p.barrier_current
        physics.short_circuit_current = r_p.short_circuit_current
        dot_spacing = np.abs(draw(r_p.dot_spacing, _rng))
        x_margins = np.abs(draw(r_p.x_margins, _rng))
        x_len = 2 * x_margins + (n_dots-1) * dot_spacing
        physics.x = np.linspace(-x_len/2, x_len/2, r_p.num_x_points, endpoint=True)
        q = draw(r_p.q, _rng)
        physics.q = q
        physics.K_0 = np.abs(draw(r_p.K_0, _rng))
        physics.sigma = np.abs(draw(r_p.sigma, _rng))
        physics.mu = draw(r_p.mu, _rng)
        physics.g_0 = np.abs(draw(r_p.g_0, _rng))
        c_k = np.abs(draw(r_p.c_k, _rng))
        physics.c_k = c_k
        scr = np.abs(draw(r_p.screening_length, _rng))
        physics.screening_length = scr
        physics.WKB_coef = np.abs(draw(r_p.WKB_coef, _rng))
        physics.v_F = np.abs(draw(r_p.v_F, _rng))
        physics.V_R = draw(r_p.V_R, _rng)
        physics.V_L = draw(r_p.V_L, _rng)
        physics.kT = np.abs(draw(r_p.kT, _rng))
        physics.beta = np.abs(draw(r_p.beta, _rng))
        h = np.abs(draw(r_p.h, _rng))
        rho = np.abs(draw(r_p.rho, _rng))
        q_scale = q if r_p.multiply_gates_by_q else 1
        gates = [simulation.GateParameters(screen=scr) for i in range(2*n_dots+1)]
        for i in range(2*n_dots+1):
            gates[i].h = np.abs(h + draw(r_p.h_rel_variations, _rng) * h)
            gates[i].rho = np.abs(rho + draw(r_p.rho_rel_variations, _rng) * rho)
            gates[i].mean = (i-n_dots)*dot_spacing/2 + draw(r_p.gate_x_rel_variations, _rng) * (dot_spacing)
        bar_peak = draw(r_p.barrier_peak, _rng)
        gates[0].peak = (bar_peak + draw(r_p.external_barrier_peak_variations, _rng)) * q_scale
        gates[2*n_dots].peak = (bar_peak + draw(r_p.external_barrier_peak_variations, _rng)) * q_scale
        for i in range(1, 2*n_dots, 2):
            gates[i].peak = (draw(r_p.plunger_peak, _rng)) * q_scale
        for i in range(2, 2*n_dots, 2):
            gates[i].peak = (bar_peak + draw(r_p.barrier_peak_variations, _rng)) * q_scale
        physics.gates = gates
        n_sens = r_p.num_sensors
        sensor_y = draw(r_p.sensor_y, _rng)
        sensors = np.zeros((n_sens, 3), dtype=np.float64)
        for i in range(n_sens):
            sensors[i] = (((i+1+draw(r_p.sensor_x_rel_variations, _rng))*x_len/(n_sens+1)-x_len/2,
                        sensor_y + draw(r_p.sensor_y_rel_variations, _rng) * sensor_y, 0))
        physics.sensors = sensors
        output.append(physics)
    return output[0] if num_physics is None else output



def calc_csd(n_dots:int, physics:simulation.PhysicsParameters,
                V_x:NDArray[np.float64], V_y:NDArray[np.float64],
                V_gates:NDArray[np.float64], x_dot:int, y_dot:int,
                include_excited:bool=True, include_converged=False) -> CSDOutput:
    '''
    Calculates a 2D charge-stability diagram, varying plunger voltages on
    2 dots and keeping all other gates constant.

    Parameters
    ----------
    n_dots : int
        The number of dots in the device.
    physics : PhysicsParameters
        The physical parameters of the device to simulate.
    V_x_min, V_x_max, N_V_x: float, float, int
        The mesh points along the x-axis will be taken from ``np.linspace(V_x_min, V_x_max, N_v_x)``.
    V_y_min, V_y_max, N_V_y: float, float, int
        The mesh points along the y-axis will be taken from ``np.linspace(V_y_min, V_y_max, N_v_y)``.
    V_gates: list[float]
        A list of length `n_dots` giving the voltages of each of the
        plunger gates (not barrier gates).
    x_dot, y_dot: int
        Integers between 0 and (n_dots - 1) inclusive, denoting the dots
        whose corresponding gate voltages are plotted on the x- or y-axes

    Returns
    -------
    CSDOutput
        A ``CSDOutput`` object wrapping the results of the computation.
    '''
    # make deep copy of physics, since gates will be modified
    phys = physics.copy()
    
    phys.K_mat = simulation.calc_K_mat(phys.x, phys.K_0, phys.sigma)
    phys.g0_dx_K_plus_1_inv = np.linalg.inv(phys.g_0*(phys.x[1]-phys.x[0])*phys.K_mat + np.identity(len(phys.x)))
    
    for d, v in enumerate(V_gates):
        phys.gates[2*d+1].peak = v

    N_v_x = len(V_x)
    N_v_y = len(V_y)

    csd_out = CSDOutput(physics=physics, V_x_vec=V_x, V_y_vec=V_y,
                        x_gate=x_dot, y_gate=y_dot, V_gates=V_gates,
                        sensor=np.zeros((N_v_x, N_v_y, len(phys.sensors)), dtype=np.float32),
                        are_dots_occupied=np.full((N_v_x, N_v_y, n_dots), False, dtype=np.bool_),
                        are_dots_combined=np.full((N_v_x, N_v_y, n_dots-1), False, dtype=np.bool_),
                        dot_charges=np.zeros((N_v_x, N_v_y, n_dots), dtype=np.int_),
                        converged=None, excited_sensor=None)
    if include_converged:
        csd_out.converged = np.full((N_v_x, N_v_y), False, dtype=np.bool_)
    if include_excited:
        csd_out.excited_sensor = np.zeros((N_v_x, N_v_y, len(phys.sensors)), dtype=np.float32)

    dot_charge = np.zeros(n_dots, dtype=np.int_)
    are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)
    ex_dot_charge = np.zeros(n_dots, dtype=np.int_)
    ex_are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)
    n_guess_prev = None
    for j in range(N_v_y):
        n_guess = n_guess_prev
        for i in range(N_v_x):
            phys.gates[2*x_dot+1].peak = V_x[i]
            phys.gates[2*y_dot+1].peak = V_y[j]
            eff_peaks = simulation.calc_effective_peaks(phys.gates)
            phys.effective_peaks = eff_peaks
            V = simulation.calc_V(phys.gates, phys.x, 0, 0, eff_peaks) 
            phys.V = V
            tf = simulation.ThomasFermi(phys)
            tf_out = tf.run_calculations(n_guess=n_guess)
            n_guess = tf.n
            csd_out.are_dots_occupied[i,j,:] = tf_out.are_dots_occupied
            csd_out.are_dots_combined[i,j,:] = tf_out.are_dots_combined
            csd_out.dot_charges[i,j,:] = tf_out.dot_charges
            csd_out.sensor[i,j,:] = tf_out.sensor
            if csd_out.converged is not None:
                csd_out.converged[i,j] = tf_out.converged
            if i == 0:
                n_guess_prev = n_guess
                if include_excited:
                    dot_charge = tf_out.dot_charges
                    are_dot_combined = tf_out.are_dots_combined
                    ex_dot_charge = dot_charge
                    ex_are_dot_combined = are_dot_combined
            if include_excited and csd_out.excited_sensor is not None:
                if np.any(simulation.is_transition(dot_charge, are_dot_combined,
                            tf_out.dot_charges, tf_out.are_dots_combined)[0]):
                    ex_dot_charge = dot_charge
                    ex_are_dot_combined = are_dot_combined
                dot_charge = tf_out.dot_charges
                are_dot_combined = tf_out.are_dots_combined    
                csd_out.excited_sensor[i,j,:] = tf.sensor_from_charge_state(ex_dot_charge, ex_are_dot_combined)

    return csd_out
    


def calc_2d_csd(physics:simulation.PhysicsParameters,
                V_x:NDArray[np.float64], V_y:NDArray[np.float64],
                include_excited:bool=True, include_converged=False) -> CSDOutput:
    '''
    Calculates a 2D charge-stability diagram, varying plunger voltages on
    2 dots and keeping all other gates constant.

    Parameters
    ----------
    physics : PhysicsParameters
        The physical parameters of the device to simulate.
    V_x_min, V_x_max, N_V_x: float, float, int
        The mesh points along the x-axis will be taken from ``np.linspace(V_x_min, V_x_max, N_v_x)``.
    V_y_min, V_y_max, N_V_y: float, float, int
        The mesh points along the y-axis will be taken from ``np.linspace(V_y_min, V_y_max, N_v_y)``.

    Returns
    -------
    CSDOutput
        A ``CSDOutput`` object wrapping the results of the computation.
    '''
    return calc_csd(2, physics, V_x, V_y, np.array([0,0]), 0, 1, include_excited=include_excited, include_converged=include_converged)



def calc_transitions(dot_charges:NDArray[np.int_], are_dots_combined:NDArray[np.bool_]) \
                    -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    '''
    Calculates the locations and types of transitions.

    A transition is defined to be present at a pixel if it has a charge state
    that varies from any of its adjacent neighbors.

    Parameters
    ----------
    dot_charges : ndarray[int]
        An array with shape ``(csd_x, [...,] n_dots)``
        indicating how many electrons are in each dot. 
        In the case of combined dots, the total number of charges should be
        entered in the left-most slot, with the other slots padded with zeros.
    are_dots_combined : ndarray[bool]
        An array with shape ``(csd_x, [...,] n_dots-1)``, 
        indicating whether the dots on either side of each barrier are combined
        together.
    
    Returns
    -------
    is_transition : ndarray[bool]
        An array with shape ``(csd_x, [...,] n_dots)`` indicating
        whether a transition is present in a particular dot. A transition occurs
        at a particular pixel and dot if the number of charges in that dot differ
        in any adjecent pixels. 
    is_transition_combined : ndarray[bool]
        An array with shape ``(csd_x, [...,] n_dots-1)``
        indicating whether there is a transition in a combined dot on either
        side of a particular barrier.  
    '''
    is_transition = np.full(dot_charges.shape, False, dtype=np.bool_)
    is_transition_combined = np.full(are_dots_combined.shape, False, dtype=np.bool_)
    for p in np.ndindex(dot_charges.shape[:-1]):
        neighbors = []
        for i in range(len(p)):
            if p[i] > 0:
                pl = list(p)
                pl[i] -= 1
                neighbors.append(tuple(pl))
            if p[i] < dot_charges.shape[i] - 1:
                pl = list(p)
                pl[i] += 1
                neighbors.append(tuple(pl))
        for nei in neighbors:
            is_tr, is_tr_com = simulation.is_transition(dot_charges[p], are_dots_combined[p], dot_charges[nei], are_dots_combined[nei])
            is_transition_combined[p] = np.logical_or(is_transition_combined[p], is_tr)
            is_transition[p] = np.logical_or(is_transition[p], is_tr_com)
    return is_transition, is_transition_combined




@dataclass(kw_only=True)
class RaysOutput:
    '''
    Output of charge stability diagram calculations. Some attributes may be ``None``
    depending on which quantities are calculated.

    Output values are stored in arrays such that the output for a property at point
    ``(V_x_vec[i], V_y_vec[j])`` is given by ``property[i][j]``.

    Attributes
    ----------
    
    physics : PhysicsParameters
        The set of physics parameters used in the simulation. 
    V_x_vec, V_y_vec : ndarray[float]
        Arrays of voltage values along the x- and y-axes.
    x_gate, y_gate : int
        The dot numbers of the dots whose gate voltages are plotted on the x- or y-axes.
    V_gates : ndarray[float]
        An array of length `n_dots` giving the voltages of each of the
        plunger gates (not barrier gates).
    sensor : ndarray[float]
        The Coulomb potential at each sensor.
    are_dots_occupied : ndarray[bool]
        An array of booleans, one for each dot, indicating whether each dot is occupied.    
    are_dots_combined : ndarray[bool]
        An array of booleans, one for each internal barrier,
        indicating whether the dots on each side are combined together
        (i.e. the barrier is too low).
    dot_charges : ndarray[int]
        An array of integers, one for each dot, indicating the total number
        of charges in each dot. In the case of combined dots, the
        total number of charges will be entered in the left-most dot,
        with the other dots padded with zeros.
    converged : ndarray[bool]
        Whether the calculation of n(x) properly converged.
    dot_transitions : ndarray[bool]
        Whether a transition is present at each dot.
    are_transitions_combined : ndarray[bool]
        Whether a combined transition is present on either side of each barrier.
    '''
    physics:simulation.PhysicsParameters=field(default_factory=lambda:simulation.PhysicsParameters())
    centers:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64)) # shape (n_centers, n_plungers)
    rays:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64)) # shape (n_rays, n_plungers)
    resolution:int=0 # must be at least 2
    sensor:NDArray[np.float32]=field(default_factory=lambda:np.zeros(0, dtype=np.float32)) # shape (n_centers, n_rays, resolution, n_sens)
    are_dots_occupied:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    are_dots_combined:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    dot_charges:NDArray[np.int_]=field(default_factory=lambda:np.zeros(0, dtype=np.int_))
    converged:NDArray[np.bool_]|None=None
    dot_transitions:NDArray[np.bool_]|None=None
    are_transitions_combined:NDArray[np.bool_]|None=None
    excited_sensor:NDArray[np.float32]|None=None
    
    def _get_physics(self) -> simulation.PhysicsParameters:
        return self._physics
    def _set_physics(self, val:simulation.PhysicsParameters):
        self._physics = val.copy()

    def _get_centers(self) -> NDArray[np.float64]:
        return self._centers
    def _set_centers(self, val:NDArray[np.float64]):
        self._centers = np.array(val, dtype=np.float64)

    def _get_rays(self) -> NDArray[np.float64]:
        return self._rays
    def _set_rays(self, val:NDArray[np.float64]):
        self._rays = np.array(val, dtype=np.float64)

    def _get_sensor(self) -> NDArray[np.float32]:
        return self._sensor
    def _set_sensor(self, val:NDArray[np.float32]):
        self._sensor = np.array(val, dtype=np.float32)

    def _get_converged(self) -> NDArray[np.bool_]|None:
        return self._converged
    def _set_converged(self, val:NDArray[np.bool_]|None):
        self._converged = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_dots_occupied(self) -> NDArray[np.bool_]:
        return self._are_dots_occupied
    def _set_are_dots_occupied(self, val:NDArray[np.bool_]):
        self._are_dots_occupied = np.array(val, dtype=np.bool_)

    def _get_are_dots_combined(self) -> NDArray[np.bool_]:
        return self._are_dots_combined
    def _set_are_dots_combined(self, val:NDArray[np.bool_]):
        self._are_dots_combined = np.array(val, dtype=np.bool_)

    def _get_dot_charges(self) -> NDArray[np.int_]:
        return self._dot_states
    def _set_dot_charges(self, val:NDArray[np.int_]):
        self._dot_states = np.array(val, dtype=np.int_)

    def _get_dot_transitions(self) -> NDArray[np.bool_]|None:
        return self._dot_transitions
    def _set_dot_transitions(self, val:NDArray[np.bool_]|None):
        self._dot_transitions = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_transitions_combined(self) -> NDArray[np.bool_]|None:
        return self._are_transitions_combined
    def _set_are_transitions_combined(self, val:NDArray[np.bool_]|None):
        self._are_transitions_combined = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_excited_sensor(self) -> NDArray[np.float32]|None:
        return self._excited_sensor
    def _set_excited_sensor(self, val:NDArray[np.float32]|None):
        self._excited_sensor = np.array(val, dtype=np.float32) if val is not None else None


    @classmethod
    def from_dict(cls, d:dict[str, Any]) -> Self:
        '''
        Creates a new ``RaysOutput`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        RaysOutput
            A new ``RaysOutput`` object with the values specified by ``dict``.
        '''
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output
    

    def to_dict(self) -> dict[str, Any]:
        '''
        Converts the ``RaysOutput`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``RaysOutput`` object.
        '''
        return dataclasses.asdict(self)
    

    def copy(self) -> Self:
        '''
        Creates a copy of a ``RaysOutput`` object.

        Returns
        -------
        RaysOutput
            A new ``RaysOutput`` object with the same attribute values as ``self``.
        '''
        return dataclasses.replace(self)

RaysOutput.physics = property(RaysOutput._get_physics, RaysOutput._set_physics) # type: ignore
RaysOutput.centers = property(RaysOutput._get_centers, RaysOutput._set_centers) # type: ignore
RaysOutput.rays = property(RaysOutput._get_rays, RaysOutput._set_rays) # type: ignore
RaysOutput.sensor = property(RaysOutput._get_sensor, RaysOutput._set_sensor) # type: ignore
RaysOutput.converged = property(RaysOutput._get_converged, RaysOutput._set_converged) # type: ignore
RaysOutput.are_dots_occupied = property(RaysOutput._get_are_dots_occupied, RaysOutput._set_are_dots_occupied) # type: ignore
RaysOutput.are_dots_combined = property(RaysOutput._get_are_dots_combined, RaysOutput._set_are_dots_combined) # type: ignore
RaysOutput.dot_charges = property(RaysOutput._get_dot_charges, RaysOutput._set_dot_charges) # type: ignore
RaysOutput.dot_transitions = property(RaysOutput._get_dot_transitions, RaysOutput._set_dot_transitions) # type: ignore
RaysOutput.are_transitions_combined = property(RaysOutput._get_are_transitions_combined, RaysOutput._set_are_transitions_combined) # type: ignore
RaysOutput.excited_sensor = property(RaysOutput._get_excited_sensor, RaysOutput._set_excited_sensor) # type: ignore



def quasirandom_points(dim, min=None, max=None, n_start=0):
    '''
        Generates quasirandom points between min and max.
        The first point returned by this generator will be initial + n_start * a,
        shifted by a multiple of (max - min) so that it falls between min and max,
        where a is the difference between consecutive points in the sequence
        prior to the shift
    '''
    # phi_n is the solution to x^(n_dim+1) - x - 1 == 0
    # see extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    phi_n = 2.
    #converges after 20 iterations
    for i in range(20): 
        phi_n = pow(1+phi_n, 1/(dim+1)) 
    
    if min is None:
        min = np.zeros(dim)
    elif hasattr(min, '__len__'):
        min = np.array(min)
    else:
        min = np.full(dim, min)
    if max is None:
        max = np.ones(dim)
    elif hasattr(max, '__len__'):
        max = np.array(max)
    else:
        max = np.full(dim, max)
   
    initial = (min + max) / 2

    s = max - min
    a = np.zeros(dim)
    phi_pow = 1.
    for i in range(dim):
        phi_pow = phi_pow / phi_n
        a[i] = s[i] * phi_pow

    x = (((initial + n_start * a - min) / s) % 1) * s + min

    while True:
        yield tuple(x)
        x = x + a
        for i in range(dim):
            if x[i] >= max[i]:
                x[i] = x[i] - s[i]



def calc_rays(physics:simulation.PhysicsParameters, centers:NDArray[np.float64],
              rays:NDArray[np.float64], resolution:int,
              include_excited:bool=True, include_converged=False) -> RaysOutput:
    '''
    Calculates a 2D charge-stability diagram, varying plunger voltages on
    2 dots and keeping all other gates constant.

    Parameters
    ----------
    physics : PhysicsParameters
        The physical parameters of the device to simulate.
    centers : NDArray[np.float64]
        shape (n_centers, n_plungers)
    rays : NDArray[np.float64]
        shape (n_rays, n_plungers)
    resolution : int
        The number of points

    Returns
    -------
    RaysOutput
        A ``RaysOutput`` object wrapping the results of the computation.
    '''
    # make deep copy of physics, since gates will be modified
    phys = physics.copy()
    
    phys.K_mat = simulation.calc_K_mat(phys.x, phys.K_0, phys.sigma)
    phys.g0_dx_K_plus_1_inv = np.linalg.inv(phys.g_0*(phys.x[1]-phys.x[0])*phys.K_mat + np.identity(len(phys.x)))
    
    n_dots = centers.shape[1]
    n_centers = centers.shape[0]
    n_rays = rays.shape[0]

    rays_out = RaysOutput(physics=physics, centers=centers, rays=rays, resolution=resolution,
                        sensor=np.zeros((n_centers, n_rays, resolution, len(phys.sensors)), dtype=np.float32),
                        are_dots_occupied=np.full((n_centers, n_rays, resolution, n_dots), False, dtype=np.bool_),
                        are_dots_combined=np.full((n_centers, n_rays, resolution, n_dots-1), False, dtype=np.bool_),
                        dot_charges=np.zeros((n_centers, n_rays, resolution, n_dots), dtype=np.int_),
                        converged=None, excited_sensor=None)
    if include_converged:
        rays_out.converged = np.full((n_centers, n_rays, resolution), False, dtype=np.bool_)
    if include_excited:
        rays_out.excited_sensor = np.zeros((n_centers, n_rays, resolution, len(phys.sensors)), dtype=np.float32)

    dot_charge = np.zeros(n_dots, dtype=np.int_)
    are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)
    ex_dot_charge = np.zeros(n_dots, dtype=np.int_)
    ex_are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)

    for c_i in range(n_centers):
        n_guess_center = None
        for r_i in range(n_rays):
            n_guess = n_guess_center
            for i in range(resolution):
                if i == 0 and r_i != 0:
                    rays_out.are_dots_occupied[c_i,r_i,0,:] = rays_out.are_dots_occupied[c_i,0,0,:]
                    rays_out.are_dots_combined[c_i,r_i,0,:] = rays_out.are_dots_combined[c_i,0,0,:]
                    rays_out.dot_charges[c_i,r_i,0,:] = rays_out.dot_charges[c_i,0,0,:]
                    rays_out.sensor[c_i,r_i,0,:] = rays_out.sensor[c_i,0,0,:]
                    if rays_out.converged is not None:
                        rays_out.converged[c_i,r_i,0] = rays_out.converged[c_i,0,0]
                    if include_excited:
                            dot_charge = rays_out.dot_charges[c_i,0,0,:]
                            are_dot_combined = rays_out.are_dots_combined[c_i,0,0,:]
                            ex_dot_charge = dot_charge
                            ex_are_dot_combined = are_dot_combined
                    if include_excited and rays_out.excited_sensor is not None:
                        rays_out.excited_sensor[c_i,r_i,0,:] = rays_out.excited_sensor[c_i,0,0,:]
                else:
                    pnt = centers[c_i] + i/(resolution-1) * rays[r_i]
                    for d_i in range(n_dots):
                        phys.gates[2*d_i+1].peak = pnt[d_i]
                    eff_peaks = simulation.calc_effective_peaks(phys.gates)
                    phys.effective_peaks = eff_peaks
                    V = simulation.calc_V(phys.gates, phys.x, 0, 0, eff_peaks) 
                    phys.V = V
                    tf = simulation.ThomasFermi(phys)
                    tf_out = tf.run_calculations(n_guess=n_guess)
                    n_guess = tf.n
                    rays_out.are_dots_occupied[c_i,r_i,i,:] = tf_out.are_dots_occupied
                    rays_out.are_dots_combined[c_i,r_i,i,:] = tf_out.are_dots_combined
                    rays_out.dot_charges[c_i,r_i,i,:] = tf_out.dot_charges
                    rays_out.sensor[c_i,r_i,i,:] = tf_out.sensor
                    if rays_out.converged is not None:
                        rays_out.converged[c_i,r_i,i] = tf_out.converged
                    if i == 0:
                        n_guess_center = n_guess
                        if include_excited:
                            dot_charge = tf_out.dot_charges
                            are_dot_combined = tf_out.are_dots_combined
                            ex_dot_charge = dot_charge
                            ex_are_dot_combined = are_dot_combined
                    if include_excited and rays_out.excited_sensor is not None:
                        if np.any(simulation.is_transition(dot_charge, are_dot_combined,
                                    tf_out.dot_charges, tf_out.are_dots_combined)[0]):
                            ex_dot_charge = dot_charge
                            ex_are_dot_combined = are_dot_combined
                        dot_charge = tf_out.dot_charges
                        are_dot_combined = tf_out.are_dots_combined    
                        rays_out.excited_sensor[c_i,r_i,i,:] = tf.sensor_from_charge_state(ex_dot_charge, ex_are_dot_combined)
    return rays_out
