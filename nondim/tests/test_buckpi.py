import sympy
from sympy.physics.units import Dimension
import sympy.physics.units as units
import sympy.physics.units.systems.si as si

from ..buckpi import nondim, pi_groups
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

    g = pi_groups([units.velocity, density, units.length, dviscosity, units.force], dimsys=dimsys)
    assert len(g) == 2
    assert u.is_dimensionless(g[0], dimsys)
    assert u.is_dimensionless(g[1], dimsys)

def test_sphere_drag():
    # Taken from "A First Course in Dimensional Analysis:
    # Simplifying Complex Phenomena Using Physical Insight" pp. 30
    dimsys, density, dviscosity = u.extend(
        ('density', 'rho', units.mass/units.volume),
        ('dviscosity', 'mu', units.pressure * units.time)
    )
    drag, b, v, rho, mu = sympy.symbols('drag b v rho mu')
    dimmap = {
        v: units.velocity,
        rho: density,
        b: units.length,
        mu: dviscosity,
        drag: units.force, 
    }
    eq = sympy.Eq(drag, sympy.Function('f')(v,rho,b,mu))
    # Cd = F(Re)
    r = nondim(eq, dimmap, dimsys=dimsys)
    assert isinstance(r, sympy.Eq)    
    assert isinstance(r.rhs, sympy.Function)

    assert drag in r.lhs.free_symbols
    assert not drag in r.rhs.free_symbols    
    assert len(r.rhs.args) == 1 # Reynolds-number (Re) or 1/Re    
    assert u.is_dimensionless(r.lhs.subs(dimmap), dimsys)
    assert u.is_dimensionless(r.rhs.args[0].subs(dimmap), dimsys)    

    Re = rho*v*b/mu
    Cd = drag/(rho*v**2*b**2)

    assert sympy.simplify(r.lhs - Cd) == 0  or sympy.simplify(r.lhs - 1/Cd) == 0
    assert sympy.simplify(r.rhs.args[0] - Re) == 0  or sympy.simplify(r.rhs.args[0] - 1/Re) == 0

def test_pendulum_period():
    # Taken from "A Student’s Guide to Dimensional Analysis" pp. ix

    t, m, l, g, theta = sympy.symbols('t m l g theta')

    dimmap = {
        t:units.time, 
        m:units.mass, 
        l:units.length, 
        g:units.acceleration, 
        theta:units.Dimension(1)
    }

    eq = sympy.Eq(t, sympy.Function('f')(m,l,g,theta))
    # sqrt(g)*t/sqrt(l) =  F(theta)
    r = nondim(eq, dimmap)

    assert isinstance(r, sympy.Eq)
    assert isinstance(r.rhs, sympy.Function)
    assert t in r.lhs.free_symbols
    assert t not in r.rhs.free_symbols
    assert m not in r.lhs.free_symbols
    assert m not in r.rhs.free_symbols

    F = r.rhs
    sr = sympy.solve(r, t)[0] # moves sqrt(g/l) to other side
    assert sympy.simplify(sr - sympy.sqrt(l)/sympy.sqrt(g)*F) == 0

    # small angle assumption
    # https://www.acs.psu.edu/drussell/Demos/Pendulum/Pendulum.html
    # https://brilliant.org/wiki/small-angle-approximation/#:~:text=The%20small%2Dangle%20approximation%20is,tan%20%E2%81%A1%20%CE%B8%20%E2%89%88%20%CE%B8%20.


def test_newton_2nd_constant():
    # Taken from "A First Course in Dimensional Analysis:
    # Simplifying Complex Phenomena Using Physical Insight" pp. 50
    xo, g, t = sympy.symbols('xo g t')
    dimmap = {
        xo:units.length, 
        g:units.acceleration, 
        t:units.time,
    }

    eq = sympy.Eq(t, sympy.Function('f')(g,xo))
    # sqrt(g)/sqrt(xo)*t = C
    r = nondim(eq, dimmap)
    assert isinstance(r, sympy.Eq)
    assert isinstance(r.lhs, sympy.Symbol)
    assert not isinstance(r.rhs, sympy.Function)
    assert r.lhs.name == 'C'
    assert sympy.simplify(r.rhs - sympy.sqrt(g)*t/sympy.sqrt(xo)) == 0

    eq = sympy.Eq(t**2, sympy.Function('f')(g,xo))
    # g*t**2/xo = C
    r = nondim(eq, dimmap)
    assert isinstance(r, sympy.Eq)
    assert isinstance(r.lhs, sympy.Symbol)
    assert not isinstance(r.rhs, sympy.Function)
    assert r.lhs.name == 'C'
    assert sympy.simplify(r.rhs - g*t**2/xo) == 0

    