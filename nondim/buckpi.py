from typing import Iterable, Tuple, List, Sequence, Mapping, Union
from collections import abc
import itertools as it
from functools import reduce

import sympy.matrices as matrices
import sympy.physics.units as units
import sympy.physics.units.systems.si as si
from sympy import Symbol

def dimensional_matrix(
        vdims: Iterable[units.Dimension], 
        dimsys: units.DimensionSystem = None) -> Tuple[matrices.Matrix]:
    '''Returns the Nd x Nv dimensional matrix.'''

    if dimsys is None:
        dimsys = si.dimsys_default

    # Form a sparse set of represented dimensions (non-zero)
    # ddict is List[Dict[str, float]]
    ddict = [dimsys.get_dimensional_dependencies(v) for v in vdims]
    basedims = list(set(it.chain(*[dd.keys() for dd in ddict])))

    # Build the Nd x Nv dimensional matrix M with Mij being the exponent
    # of the j-th variable in the i-th base dimension.
    Nv = len(vdims)
    Nd = len(basedims)
    dm = matrices.zeros(Nd, Nv)
    for vid, dd in enumerate(ddict):
        for d, e in dd.items():
            dm[basedims.index(d), vid] = e

    return dm

Variables = Union[Mapping[Symbol, units.Dimension], Iterable[units.Dimension]]

def pi_groups(
        variables: Variables, 
        dimsys: units.DimensionSystem = None):
    '''Returns all independent variable products that non-dimensionalize the given variables.'''

    if len(variables) == 0:
        raise ValueError('Need at least one variable.')

    vsyms = None
    vdims = None
    if isinstance(variables, abc.Mapping):
        vsyms = variables.keys()
        vdims = variables.values()
    else:
        vdims = list(variables)
        vsyms = vdims
        
    # The nullity of the dimensional matrix {v|Av=0} represents all possible independent variable product groups, with v_i being the exponent of the i-th variable.
    dm = dimensional_matrix(vdims, dimsys=dimsys)
    nullity = dm.nullspace()
    groups = []
    for nv in nullity:
        generator = zip(vsyms, nv)
        first = next(generator)
        pi = reduce(
            lambda t,ve: t * (ve[0]**ve[1]), 
            generator, 
            first[0]**first[1]
        )
        groups.append(pi)
    return groups

# Function alias
nondim = pi_groups


if __name__ == '__main__':
    import sympy
    var = [units.force, units.time, units.length, units.mass]
    a,b,c,d = sympy.symbols('a b c d')

    print(pi_groups(var, dimsys=si.dimsys_SI))
    print(pi_groups({a:units.force, b:units.time, c:units.length, d:units.mass}, dimsys=si.dimsys_SI))

    g = pi_groups({a:units.force, b:units.time, c:units.length, d:units.mass}, dimsys=si.dimsys_SI)

    print(sympy.latex(g))

    from .utils import extend
    dimsys, (density, dviscosity) = extend(
        ('density', 'rho', units.mass/units.volume),
        ('dynamic_viscosity', 'mu', units.pressure*units.time)
    )
    print(sympy.latex(pi_groups([
        units.force,
        units.length,
        units.velocity,
        density,
        dviscosity
    ], dimsys=dimsys)))

