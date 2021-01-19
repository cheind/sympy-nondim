import sympy
import sympy.physics.units as units
import sympy.physics.units.systems.si as si

from ..buckpi import nondim
from .. import utils as u

def test_nondim():
    g = nondim([units.mass, units.length])
    assert len(g) == 0

    g = nondim([units.mass, units.mass])
    assert len(g) == 1
    assert u.is_dimensionless(g[0])

    dimsys, density, dviscosity = u.extend(
        ('density', 'rho', units.mass/units.volume),
        ('dviscosity', 'mu', units.pressure * units.time)
    )
    drag,b,v,rho,mu = sympy.symbols('drag b v rho mu')

    # The order of variables is kind-of important:
    # Move the variables that you want to appear only in one of the terms 
    # towards the end. 
    # TODO: specify the dependent variable explicitly and ensure that
    # it appears only in one of the terms (which might not be always possible). If the nullspace is n-dimensional, than this is a nxn matrix G whose columns represent gi represent the scalar factor combinations of the nullspace vectors. That matrix has to have full-rank or the transformed null space will have less degrees of freedom.
    sdict = {          
        v:units.velocity,
        rho:density,        
        b:units.length,
        mu:dviscosity,               
        drag:units.force,
    }
    gs = nondim(sdict, dimsys=dimsys)
    assert len(gs) == 2 # reynolds number and drag-coefficient (without the constant)
    assert u.is_dimensionless(gs[0].subs(sdict), dimsys)
    assert u.is_dimensionless(gs[1].subs(sdict), dimsys)
    assert sum([drag in g.free_symbols for g in gs]) == 1 # drag force appears only in one of the expr. (cd)
    assert sum([mu in g.free_symbols for g in gs]) == 1 # dyn. viscosity appears only in one of the expr. (re)
    re = rho*v*b/mu
    cd = drag/(rho*v**2*b**2)
    # In the following note,
    # a. [v] == [1/v] for dimless v
    # b. constants c are not (like 0.5 in cd) are not provided by dim. analysis
    # That is
    #   y = F1(v) is eq. to y = F2(1/v) or 1/y = F3(1/v)
    #   cy = F(v) is eq. to y = F2(v) where the constant moved into F2
    assert sum([re==g or 1/re==g for g in gs]) == 1 # one over is fine also since [v] == [1/v] for dimensionless variables
    assert sum([cd==g or 1/cd==g for g in gs]) == 1 # one over is fine also since [v] == [1/v] for dimensionless variables

    