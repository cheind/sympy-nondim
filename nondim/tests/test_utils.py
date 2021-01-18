import sympy.physics.units as units
import sympy.physics.units.systems.si as si
from sympy import symbols

from .. import utils as u


def test_extend():
    dimsys, density = u.extend(('density', 'rho', units.mass/units.volume))
    assert dimsys.get_dimensional_dependencies(
        density) == {'mass': 1, 'length': -3}
    assert density.name == symbols('density')
    assert density.symbol == symbols('rho')


def test_dimensionless():
    assert not u.is_dimensionless(units.mass)
    assert u.is_dimensionless(units.mass / units.mass)
    dimsys, density = u.extend(('density', 'rho', units.mass/units.volume))
    assert not u.is_dimensionless(density, dimsys)
