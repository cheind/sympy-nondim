
import sympy
from sympy import matrices
from sympy import symbols
from sympy.physics.units.systems import SI
from sympy.physics.units import length, mass, acceleration, force, velocity,volume, pressure, time, Dimension
from sympy.physics.units import gravitational_constant as G
from sympy.physics.units.systems.si import dimsys_SI
import numpy as np

import itertools as it

def pi_groups(*variables):
    # Create dimensional matrix
    dimdicts = [
        dimsys_SI.get_dimensional_dependencies(v) 
        for v in variables
    ]
    
    base_dims = list(set(it.chain(*[dd.keys() for dd in dimdicts])))

    Nv = len(variables)
    Nd = len(base_dims)
    dm = matrices.zeros(Nd, Nv)
    for cid, dd in enumerate(dimdicts):
        for d,v in dd.items():
            dm[base_dims.index(d), cid] = v

    nullity = dm.nullspace()
    groups = []
    for nv in nullity:
        pi = Dimension(1)
        for v,e in zip(variables, nv):
            pi = pi * v**e
        groups.append(pi)
    return groups

# density = Dimension('density', symbol='rho')
# print(dimsys_SI.get_dimensional_dependencies(density))
#print(dimsys_SI.get_dimensional_dependencies(F))


groups = pi_groups(force, length, velocity, mass/volume, pressure*time)
print(groups)
print([g.is_dimensionless for g in groups])

#mass/volume, )



        # p.drag = si.F
        # p.mu = si.DynamicViscosity     
        # p.b = si.L
        # p.V = si.Velocity
        # p.rho = si.Density
        

# F = mass*acceleration
# F
# Dimension(acceleration*mass)
# dimsys_SI.get_dimensional_dependencies(F)
# {'length': 1, 'mass': 1, 'time': -2}
# dimsys_SI.get_dimensional_dependencies(force)
# {'length': 1, 'mass': 1, 'time': -2}