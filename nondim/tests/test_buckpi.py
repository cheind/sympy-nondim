import sympy.physics.units as units
import sympy.physics.units.systems.si as si

from ..buckpi import pi_groups
from .. import utils as u


def test_pi_groups():
    g = pi_groups([units.mass, units.length])
    assert len(g) == 0

    g = pi_groups([units.mass, units.mass])
    assert len(g) == 1
    assert u.is_dimensionless(g[0])

    dimsys, density, dviscosity = u.extend(
        ('density', 'rho', units.mass/units.volume),
        ('dviscosity', 'mu', units.pressure * units.time)
    )
    g = pi_groups([
        units.force,
        units.length,
        units.velocity,
        density,
        dviscosity], dimsys=dimsys)
    print(g)


"""
var = [units.force, units.time, units.length, units.mass]
a, b, c, d = sympy.symbols('a b c d')

print(pi_groups(var, dimsys=si.dimsys_SI))
print(pi_groups(
    {
        a: units.force,
        b: units.time,
        c: units.length,
        d: units.mass
    },
    dimsys=si.dimsys_SI))

g = pi_groups(
    {
        a: units.force,
        b: units.time,
        c: units.length,
        d: units.mass
    },
    dimsys=si.dimsys_SI)

# print(sympy.latex(g))

dimsys, (density, dviscosity) = extend(
    ('density', 'rho', units.mass / units.volume),
    ('dynamic_viscosity', 'mu', units.pressure * units.time))
print(
    sympy.latex(
        pi_groups([
            units.force, units.length, units.velocity, density, dviscosity
        ],
            dimsys=dimsys)))
"""
