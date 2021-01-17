import sympy.matrices as matrices
import sympy.physics.units as units
import sympy.physics.units.systems.si as si
from typing import Iterable, Tuple, List
import itertools as it
from functools import reduce

def dimensional_matrix(
        variables: Iterable[units.Dimension], 
        dimsys: units.DimensionSystem = None) -> Tuple[matrices.Matrix]:

    if dimsys is None:
        dimsys = si.dimsys_SI

    # Form a sparse set of represented dimensions (non-zero)
    # ddict is List[Dict[str, float]]
    ddict = [dimsys.get_dimensional_dependencies(v) for v in variables]
    basedims = list(set(it.chain(*[dd.keys() for dd in ddict])))

    # Build the Nd x Nv dimensional matrix M with Mij being the exponent
    # of the j-th variable in the i-th base dimension.
    Nv = len(variables)
    Nd = len(basedims)
    dm = matrices.zeros(Nd, Nv)
    for vid, dd in enumerate(ddict):
        for d, e in dd.items():
            dm[basedims.index(d), vid] = e

    return dm    

def pi_groups(variables, dimsys: units.DimensionSystem = None):
    '''Returns all independent variable products that non-dimensionalize the given variables.'''

    dm = dimensional_matrix(variables, dimsys=dimsys)

    # The nullity of the dimensional matrix {v|Av=0} represents all possible independent variable product groups with v_i being the exponent of the i-th variable.
    nullity = dm.nullspace()
    groups = []
    for nv in nullity:
        pi = reduce(
            lambda t,ve: t * (ve[0]**ve[1]), 
            zip(variables,nv), 
            units.Dimension(1)
        )
        groups.append(pi)
    return groups


if __name__ == '__main__':
    var = [units.force, units.time, units.length, units.mass]
    print(pi_groups(var, dimsys=si.dimsys_SI))