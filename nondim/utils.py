from typing import Sequence, Tuple

from sympy.physics.units import Dimension, DimensionSystem
from sympy.physics.units.systems.si import dimsys_default


def extend(*args: Tuple[str, str, Dimension], dimsys: DimensionSystem = None) -> Tuple[DimensionSystem, Sequence[Dimension]]:
    '''Extends a dimension system by the given derived dimensions.

    Registering new derived dimensions is useful in simplifying expressions
    envolving these dimensions. Take density, which we could define simply by
        density = si.mass/si.volume,
    and use it in `pi_groups(density,...)`. However, the result will be less 
    readable than registering density as derived dimension
        density, newdimsys = extend(('density','rho',si.mass/si.volume),...)

    Params
    ------
    args: Tuple[str, str, Dimension]
        Sequence of derived dimensions to add. The first argument is the name,
        the second a symbol (or None) and the last argument is the derived dimension.
    dimsys: DimensionSystem, optional
        Dimension system to extend. If not specified, extends the default system.

    Returns
    -------
    dimsys: DimensionSystem
        Extended dimension system
    dims: Sequence[Dimension]
        Derived dimensions
    '''
    if dimsys is None:
        dimsys = dimsys_default
    deps = dimsys.get_dimensional_dependencies
    dims = [Dimension(dd[0], dd[1]) for dd in args]
    depsdict = {dim: deps(dd[2]) for dim, dd in zip(dims, args)}
    results = [dimsys.extend([], new_dim_deps=depsdict)] + dims
    return results


def is_dimensionless(dim: Dimension, dimsys: DimensionSystem = None) -> bool:
    '''Tests if the given dimension is dimensionless.'''
    if dimsys is None:
        dimsys = dimsys_default
    return len(dimsys.get_dimensional_dependencies(dim)) == 0
