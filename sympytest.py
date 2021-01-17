
import sympy
from sympy import matrices
from sympy import symbols
from sympy.physics.units.systems import SI
from sympy.physics.units import length, mass, acceleration, force, velocity,volume, pressure, time, Dimension, DimensionSystem
from sympy.physics.units import gravitational_constant as G
from sympy.physics.units.systems.si import dimsys_SI
import numpy as np

import itertools as it

def pi_groups(*variables, dimsys=None):
    # Create dimensional matrix
    if dimsys is None:
        dimsys = dimsys_SI
    dimdicts = [
        dimsys.get_dimensional_dependencies(v) 
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

from dataclasses import dataclass
class DerivedDim:
    name: str
    symbol: str = None
    dims: Dimension
    
from typing import Iterable, Tuple

def extend(dds: Iterable[DerivedDim], dimsys: DimensionSystem):
    deps = dimsys.get_dimensional_dependencies
    added = {
        Dimension(d.name, d.symbol) : deps(d.dims)
        for d in dds        
    }
    return dimsys.extend([], new_dim_deps=added)

def extend(dimsys: DimensionSystem, dd: Tuple[str,str,Dimension]):
    deps = dimsys.get_dimensional_dependencies
    dim = Dimension(dd[0], dd[1])
    edimsys = dimsys.extend([], new_dim_deps={dim:deps(dd[2])})
    return edimsys, dim

class Extender:
    def __init__(self, dimsys: DimensionSystem):
        self.dimsys = dimsys
        self.edict = {}

    def __call__(
            self, 
            name: str, 
            ddim: Dimension, 
            symbol: str = None) -> Dimension:
        d = Dimension(name, symbol=symbol)
        self.edict[d] = self.dimsys.get_dimensional_dependencies(ddim)
        return d

    def apply(self):
        return self.dimsys.extend([], new_dim_deps=self.edict)

e = Extender(dimsys_SI)
density = e('density', mass/volume, symbol='rho')
dimsys = e.apply()

def is_dimless(dimsys: DimensionSystem, dim: Dimension):
    return len(dimsys.get_dimensional_dependencies(dim)) == 0



groups = pi_groups(force, length, velocity, density, pressure*time, dimsys=dimsys)
print(groups)
print([is_dimless(dimsys, g) for g in groups])

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