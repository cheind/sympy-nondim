import itertools as it
from collections import abc
from functools import reduce
from typing import Iterable, List, Mapping, Sequence, Tuple, Union

import sympy.matrices as matrices
import sympy.physics.units as units
import sympy.physics.units.systems.si as si
from sympy import Symbol, Expr, Eq, Function


def _dimensional_matrix(
        sdims: Sequence[units.Dimension],
        dimsys: units.DimensionSystem = None) -> matrices.Matrix:
    '''Returns the dimensional matrix from the given variables.

    The dimensional matrix `M` represents a `Nd (number of base dimensions) x Nv (number of variables) matrix`, whose entry `ij` corresponds to the
    exponent of the j-th variable in the i-th dimension.

    Params
    ------
    sdims: Sequence[units.Dimension]
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
    ddict = [dimsys.get_dimensional_dependencies(v) for v in sdims]
    basedims = list(set(it.chain(*[dd.keys() for dd in ddict])))

    # Build the Nd x Nv dimensional matrix M with Mij being the exponent
    # of the j-th variable in the i-th base dimension.
    Nv = len(sdims)
    Nd = len(basedims)
    dm = matrices.zeros(Nd, Nv)
    for vid, dd in enumerate(ddict):
        for d, e in dd.items():
            dm[basedims.index(d), vid] = e

    return dm

def pi_groups(
        sdims: Sequence[units.Dimension],
        syms: Sequence[Symbol] = None,
        dimsys: units.DimensionSystem = None
    ) -> Sequence[Expr]:

    # Form the dimensional matrix from the given symbol dimensions
    dm = _dimensional_matrix(sdims, dimsys=dimsys)

    # Solve for the nullspace
    nspace = dm.nullspace()

    if syms is None:
        syms = sdims

    # Build groups of variables (expr)
    # When symbols are specified this builds a sequence
    # of expression consisting of symbols, otherwise
    # Dimensions in sdims are used.
    groups = []
    for nv in nspace:
        generator = zip(syms, nv)
        first = next(generator)
        pi = reduce(lambda t, ve: t * (ve[0]**ve[1]), generator,
                    first[0]**first[1])
        groups.append(pi)
    return groups



def nondim(
        eq: Eq, 
        dimmap: Mapping[Symbol, units.Dimension],
        dimsys: units.DimensionSystem = None) -> Expr:
    '''Computes a dimensionless form of the given equation.

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
    eq: sympy.Eq
        Equation that relates the dependent variable to an unspecified sympy.Function of independent variables.
    dimmap: Mapping[Symbol, units.Dimension]
        A mapping from symbols in `eq` to their corresponding sympy.physics.units.Dimension
    dimsys: sympy.physics.units.DimensionSystem, optional
        The associated sympy.physics.units.DimensionSystem. If not specified,
        `dimsys_default` is used.

    Raises
    ------
    ValueError:
        When the number of independent dimensionless products is zero.

    Returns
    -------
    expr: sympy.Expr
        The solution of dimensional analysis, depending on the number N of dimensionless products {gi}_i<=N
            - N=0: see Raises
            - N=1: sympy.Eq(g0,C) where C is an unknown constant
            - N>1 & dep. variable appears only in gj: sympy.Eq(gj, sympy.Function({gi|i!=j}))
            - otherwise: sympy.Function({gi})
    '''

    # Evaluate the lhs/rhs dimensions
    elhs = eq.lhs.subs(dimmap)
    erhs = eq.rhs.subs(dimmap)

    # Create dimensional matrix, treating the lhs as the dependent dimension
    syms = [*eq.rhs.args] + [eq.lhs]
    sdims = [*erhs.args] + [elhs]

    # Compute lin. independent dimensionless products
    groups = pi_groups(sdims, syms, dimsys)
    
    # Determine result from groups
    ng = len(groups)
    if ng == 0:
        raise ValueError('No relation could be found.')
    elif ng == 1:
        return Eq(Symbol('C', constant=True), groups[0])
    else:
        # Search for first term that includes the original dependent variable
        deps = [g for g in groups if eq.lhs in g.free_symbols]
        indeps = [g for g in groups if eq.lhs not in g.free_symbols]
        if len(deps) == 0 or len(deps) > 1:
            return Function('F')(*deps, *indeps)
        else:
            return Eq(deps[0], Function('F')(*indeps))