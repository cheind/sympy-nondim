import itertools as it
from collections import abc
from functools import reduce
from typing import Iterable, List, Mapping, Sequence, Tuple, Union

import sympy.matrices as matrices
import sympy.physics.units as units
import sympy.physics.units.systems.si as si
from sympy import Symbol, Expr


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


def pi_groups(variables: Variables,
              dimsys: units.DimensionSystem = None) -> Sequence[Expr]:
    '''Returns all independent dimensionless variable products.

    This method is based on the Buckingham-Pi theoreom and treats non-dimensionalization in terms of linear algebra.

    Background
    ----------

    First note, physical dimensions form an abelian group unter multiplication. Each physical dimension may be represented by a vector `v` in a d-dimensional vector space, spanned by the unit vectors of base dimensions. The components of `v` represent the exponents of the corresponding base dimensions. For example, the dimension `L*M*T**-2` can be represented in by a vector whose components are [1,1,-2] in a coordinate frame spanned by the unit vectors eL, eM, eT. The zero vector [0,0,0] represents a unit/dimensionless vector.

    Consider two dimensional physical variables, `x` and `y` and let
        dim(x) = L^xl M^xm T^xt = [xl, xm, xt] =: vx
        dim(y) = L^yl M^ym T^yt = [yl, ym, yt] =: vy
    then
        dim(x*y) = L^(xl+yl) M^(xm+ym) T^(xt+yt) = vx + vy.

    Also note, for scalar `s`
        dim(x^s) = L^(s*xl) M^(s*xm) T^(s*xt) = _vx*s

    Method
    ------

    From the above follows, that the product of variables raised to specific powers can be expressed as matrix-vector product
        dim(x^s*y^t) = M*[s,t]^T,
    where `M` is the dimensional `Nd (#base-dims) x Nv (#variables)` matrix formed by column-stacking the dimensional vectors associated with each variable. Now, the result of this method are independent dimensionless variable products that are given by the set of vectors `n` for which
        {n | Mn = 0}.
    I.e the linear independent vectors spanning the nullspace of M.

    Params
    ------
    variables: Variables
        Either a sequence of sympy.physics.units.Dimension or a map from symbols to sympy.physics.units.Dimension.
    dimsys: units.DimensionSystem, optional
        The associated sympy.physics.units.DimensionSystem. If not specified,
        `dimsys_default` is used.

    Returns
    -------
    pi: Sequence[Expr]
        Sequence of expressions representing independent variable products.
    '''
    vsyms = None
    vdims = None
    if isinstance(variables, abc.Mapping):
        vsyms = variables.keys()
        vdims = variables.values()
    elif isinstance(variables, abc.Sequence):
        vdims = list(variables)
        vsyms = vdims
    else:
        raise ValueError(
            'Variables argument needs to be Mapping or Sequence type.')

    if len(variables) == 0:
        raise ValueError('Need at least one variable.')

    # The nullity of the dimensional matrix {v|Av=0} represents all possible
    # independent variable product groups, with v_i being the exponent of the
    # i-th variable.
    dm = dimensional_matrix(vdims, dimsys=dimsys)
    nspace = dm.nullspace()
    groups = []
    for nv in nspace:
        generator = zip(vsyms, nv)
        first = next(generator)
        pi = reduce(lambda t, ve: t * (ve[0]**ve[1]), generator,
                    first[0]**first[1])
        groups.append(pi)
    return groups


# Function alias
nondimensionalize = pi_groups

if __name__ == '__main__':
    import sympy
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

    from .utils import extend
    dimsys, (density, dviscosity) = extend(
        ('density', 'rho', units.mass / units.volume),
        ('dynamic_viscosity', 'mu', units.pressure * units.time))
    print(
        sympy.latex(
            pi_groups([
                units.force, units.length, units.velocity, density, dviscosity
            ],
                dimsys=dimsys)))
