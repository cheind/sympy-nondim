import sympy
from sympy.physics.units import Dimension
import sympy.physics.units as units
import sympy.physics.units.systems.si as si

from ..buckpi import nondim, nondim_eq, nondim_eq2
from .. import utils as u


def test_nondim():
    g = nondim([units.mass, units.length])
    assert len(g) == 0

    g = nondim([units.mass, units.mass])
    assert len(g) == 1
    assert u.is_dimensionless(g[0])


def test_sphere_drag():
    # Taken from "A First Course in Dimensional Analysis:
    # Simplifying Complex Phenomena Using Physical Insight"

    dimsys, density, dviscosity = u.extend(
        ('density', 'rho', units.mass/units.volume),
        ('dviscosity', 'mu', units.pressure * units.time)
    )
    drag, b, v, rho, mu = sympy.symbols('drag b v rho mu')

    # The order of variables is kind-of important:
    # Move the variables that you want to appear only in one of the terms
    # towards the end.
    # TODO: specify the dependent variable explicitly and ensure that
    # it appears only in one of the terms (which might not be always possible). If the nullspace is n-dimensional, than this is a nxn matrix G whose columns represent gi represent the scalar factor combinations of the nullspace vectors. That matrix has to have full-rank or the transformed null space will have less degrees of freedom.
    sdict = {
        v: units.velocity,
        rho: density,
        b: units.length,
        mu: dviscosity,
        drag: units.force,
    }
    gs = nondim(sdict, dimsys=dimsys)
    # reynolds number and drag-coefficient (without the constant)
    assert len(gs) == 2
    assert u.is_dimensionless(gs[0].subs(sdict), dimsys)
    assert u.is_dimensionless(gs[1].subs(sdict), dimsys)
    # drag force appears only in one of the expr. (cd)
    assert sum([drag in g.free_symbols for g in gs]) == 1
    # dyn. viscosity appears only in one of the expr. (re)
    assert sum([mu in g.free_symbols for g in gs]) == 1
    re = rho*v*b/mu
    cd = drag/(rho*v**2*b**2)
    # In the following note,
    # a. [v] == [1/v] for dimless v
    # b. constants c are not (like 0.5 in cd) are not provided by dim. analysis
    # That is
    #   y = F1(v) is eq. to y = F2(1/v) or 1/y = F3(1/v)
    #   cy = F(v) is eq. to y = F2(v) where the constant moved into F2
    # one over is fine also since [v] == [1/v] for dimensionless variables
    assert sum([re == g or 1/re == g for g in gs]) == 1
    # one over is fine also since [v] == [1/v] for dimensionless variables
    assert sum([cd == g or 1/cd == g for g in gs]) == 1


def test_pendulum_swing():
    # Taken from "A Student’s Guide to Dimensional Analysis" pp. ix

    m, l, g, t = sympy.symbols('m l g t', real=True, positive=True)
    theta = sympy.symbols('theta', real=True)

    sdict = {
        # its mass (actually will turn out to be superfluous)
        m: units.mass,
        l: units.length,         # length of pendulum
        g: units.acceleration,   # accel of gravity
        theta: Dimension(1),     # max angle
        t: units.time            # period
    }
    gs = nondim(sdict)
    f = sympy.Function('f')(gs[0])
    eq = sympy.Eq(gs[1], f)
    seq = sympy.Eq(t, sympy.solve(eq, t)[0])

    assert seq.lhs == t
    assert not m in seq.rhs.free_symbols
    assert sympy.expand(seq.rhs - sympy.sqrt(l/g)*f) == 0

    # small angle assumption
    # https://www.acs.psu.edu/drussell/Demos/Pendulum/Pendulum.html
    # https://brilliant.org/wiki/small-angle-approximation/#:~:text=The%20small%2Dangle%20approximation%20is,tan%20%E2%81%A1%20%CE%B8%20%E2%89%88%20%CE%B8%20.


def test_pendulum_swing_eq():
    # Taken from "A Student’s Guide to Dimensional Analysis" pp. ix

    # m,l,g,t = sympy.symbols('m l g t', real=True, positive=True)
    # theta = sympy.symbols('theta', real=True)

    # sdict = {
    #     m:units.mass,           # its mass (actually will turn out to be superfluous)
    #     l:units.length,         # length of pendulum
    #     g:units.acceleration,   # accel of gravity
    #     theta:Dimension(1),     # max angle
    #     t:units.time            # period
    # }
    # gs = nondim(sdict)
    # f = sympy.Function('f')(gs[0])
    # eq = sympy.Eq(gs[1], f)
    # print(eq)
    # seq = sympy.Eq(t, sympy.solve(eq, t)[0])

    # assert seq.lhs == t
    # assert not m in seq.rhs.free_symbols
    # assert sympy.expand(seq.rhs - sympy.sqrt(l/g)*f) == 0

    eq = sympy.Eq(units.time, sympy.Function('f')(
        units.mass, units.length, units.acceleration, units.Dimension(1, 'theta')))

    print(units.time.name in eq.lhs.free_symbols)
    print(nondim_eq(eq))
    print(sympy.solve(nondim_eq(eq), units.time.name)[0])

    # small angle assumption
    # https://www.acs.psu.edu/drussell/Demos/Pendulum/Pendulum.html
    # https://brilliant.org/wiki/small-angle-approximation/#:~:text=The%20small%2Dangle%20approximation%20is,tan%20%E2%81%A1%20%CE%B8%20%E2%89%88%20%CE%B8%20.


def test_pendulum_swing_eq2():
    # Taken from "A Student’s Guide to Dimensional Analysis" pp. ix

    # m,l,g,t = sympy.symbols('m l g t', real=True, positive=True)
    # theta = sympy.symbols('theta', real=True)

    # sdict = {
    #     m:units.mass,           # its mass (actually will turn out to be superfluous)
    #     l:units.length,         # length of pendulum
    #     g:units.acceleration,   # accel of gravity
    #     theta:Dimension(1),     # max angle
    #     t:units.time            # period
    # }
    # gs = nondim(sdict)
    # f = sympy.Function('f')(gs[0])
    # eq = sympy.Eq(gs[1], f)
    # print(eq)
    # seq = sympy.Eq(t, sympy.solve(eq, t)[0])

    # assert seq.lhs == t
    # assert not m in seq.rhs.free_symbols
    # assert sympy.expand(seq.rhs - sympy.sqrt(l/g)*f) == 0

    t,m,l,g,theta = sympy.symbols('t m l g theta')
    eq = sympy.Eq(t, sympy.Function('f')(m,l,g,theta))
    r = nondim_eq2(eq, {t:units.time, m:units.mass, l:units.length, g:units.acceleration, theta:units.Dimension(1)})
    print(r)
    print(r.subs({t:units.time, m:units.mass, l:units.length, g:units.acceleration, theta:units.Dimension(1)}))

    # eq = sympy.Eq(units.time, sympy.Function('f')(
    #     units.mass, units.length, units.acceleration, units.Dimension(1, 'theta')))

    # print(units.time.name in eq.lhs.free_symbols)
    # print(nondim_eq(eq))
    # print(sympy.solve(nondim_eq(eq), units.time.name)[0])

    # small angle assumption
    # https://www.acs.psu.edu/drussell/Demos/Pendulum/Pendulum.html
    # https://brilliant.org/wiki/small-angle-approximation/#:~:text=The%20small%2Dangle%20approximation%20is,tan%20%E2%81%A1%20%CE%B8%20%E2%89%88%20%CE%B8%20.
