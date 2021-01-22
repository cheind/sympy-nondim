import pytest
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

def test_no_relation():
    a,b = sympy.symbols('a b')
    with pytest.raises(ValueError):
        nondim(sympy.Eq(a,sympy.Function('f')(b)), {a:units.time, b:units.length})
        
def test_dependent_irrelevant():
    a,b,c,d = sympy.symbols('a b c d')
    r = nondim(
        sympy.Eq(a,sympy.Function('f')(b,c,d)), 
        {a:units.time, b:units.length, c:units.length, d:Dimension(1)}
    )
    assert isinstance(r, sympy.Function)
    assert any([
        r.args == (b/c, d),
        r.args == (c/b, d),
        r.args == (d, b/c),
        r.args == (d, c/b)])

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
    assert isinstance(r.rhs, sympy.Symbol)
    assert not isinstance(r.lhs, sympy.Function)
    assert r.rhs.name == 'C'
    assert sympy.simplify(r.lhs - sympy.sqrt(g)*t/sympy.sqrt(xo)) == 0

    eq = sympy.Eq(t**2, sympy.Function('f')(g,xo))
    # g*t**2/xo = C
    r = nondim(eq, dimmap)
    assert isinstance(r, sympy.Eq)
    assert isinstance(r.rhs, sympy.Symbol)
    assert not isinstance(r.lhs, sympy.Function)
    assert r.rhs.name == 'C'
    assert sympy.simplify(r.lhs - g*t**2/xo) == 0


def test_gas_pressure():
    # Taken from A Student’s Guide to Dimensional Analysis pp. 17
    # Main point here is that allow enter weight as m*g into dim. analysis rather than m,g 
    m, g, p, A = sympy.symbols('m g p A')
    dimmap = {
        m: units.mass,
        g: units.acceleration,
        p: units.pressure,
        A: units.length**2
    }

    # p = f(A,mg)
    eq = sympy.Eq(p, sympy.Function('f')(A, m*g))
    # A*p/(g*m) = C    
    r = nondim(eq, dimmap)
    assert isinstance(r, sympy.Eq)
    assert p in r.lhs.free_symbols
    assert sympy.simplify(sympy.solve(r, p)[0] - r.rhs*(m*g)/A) == 0 # rhs is C

    # Note, in this case nondim returns the same result even if m,g are passed seperately.
    # eq = sympy.Eq(p, sympy.Function('f')(A, m, g))
    # r = nondim(eq, dimmap)
