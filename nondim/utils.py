from typing import Sequence, Tuple
from sympy.physics.units import DimensionSystem, Dimension
from sympy.physics.units.systems.si import dimsys_SI

def extend(
        *args: Tuple[str,str,Dimension], 
        dimsys: DimensionSystem=None):
    if dimsys is None:
        dimsys = dimsys_SI
    deps = dimsys.get_dimensional_dependencies
    dims = [Dimension(dd[0], dd[1]) for dd in args]
    depsdict = {dim:deps(dd[2]) for dim,dd in zip(dims,args)}
    return dimsys.extend([], new_dim_deps=depsdict), dims