from typing import Iterable, Tuple, List, Sequence, Mapping, Union
from collections import abc
import itertools as it
from functools import reduce

import sympy.matrices as matrices
import sympy.physics.units as units
import sympy.physics.units.systems.si as si
from sympy import Symbol

def dimensional_matrix(
        vdims: Sequence[units.Dimension], 
        dimsys: units.DimensionSystem = None) -> matrices.Matrix:
    '''Returns the dimensional matrix from the given variables.

    The dimensional matrix `M` represents a `Nd (number of base dimensions) x Nv (number of variables) matrix`, whose entry `ij` corresponds to the
    exponent of the j-th variable in the i-th dimension.

    Params
    ------
    vdims: Sequence[units.Dimension]
        A sequence of sympy.physics.units.Dimension representing the possibly
        (derived) dimensions of each variable.
    dimsys: units.DimensionSystem, optional
        The associated sympy.physics.units.DimensionSystem. If not specified,
        `dimsys_default` is used.

    Returns
    -------
    matrix: sympy.matrices.Matrix
        Nd x Nv dimensional matrix
    '''

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

Variables = Union[Mapping[Symbol, units.Dimension], Sequence[units.Dimension]]

def pi_groups(
        variables: Variables, 
        dimsys: units.DimensionSystem = None) -> Sequence[sympy.Expr]:
    '''Returns all independent dimensionless variable products.

    This method is based on the Buckingham-Pi theoreom and frames non-dimensionalization in linear algebra terms. 
    
    First note, all derived dimensions are products of base dimensions (A,B,C) raised to some power (a,b,c):
        [x] = A^a*B^b*C^c
    The dimensions form an abelian group under multiplication and we may associate with each dimension an (exponent) vector in R^d space, whose entries (a,b,c) correspond to the exponents along the base dimensions (A,B,C) by the. The zero-vector represents
        A^0*B^0*C^0 = 1
    unity/dimensionless variables. Next, note that vector addition `x+y` corresponds to a dimensional product
        [v*w] = A^(xa+ya)B^(xb+yb)C^(xc+yc)
    Moreover, vector scalar multiplication `sx` corresponds to raising a dimension to the power s
        [v^s] = A^sxaB^sxbC^sxc
    Finally, note the product of n variables (x,y) raised to powers (s,t) can be written as matrix-vector product
        [x^s*y^t] = M*[s,t]^T
    where M is the dimensional matrix formed by column-stacking the dimensional vectors of all system variables. Then, the set of (exponent) vectors 
        {v|Mv=0, <v_i,v_j>=0} 
    naturally represents all possible independent dimensionless variable products. This set of vectors is known as the nullspace/nullity of M and represents the solution this method computes.

    Params
    ------
    variables: Variables
        Either a sequence of sympy.physics.units.Dimension or a map from symbols to sympy.physics.units.Dimension.
    dimsys: units.DimensionSystem, optional
        The associated sympy.physics.units.DimensionSystem. If not specified,
        `dimsys_default` is used.

    Returns
    pi: Sequence[sympy.Expr]
        Sequence of expressions representing independent variable products.
    '''
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

