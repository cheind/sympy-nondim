from __future__ import annotations 
import string
from typing import Union, Iterable, Optional, Iterator, List
import numpy as np

from . import utils as u

def _fmt_dimension(exponent: float, dimension_name: str) -> str:
    if np.allclose(exponent, 1): # misuse of allclose for scalars
        return dimension_name
    elif not np.allclose(exponent, 0):
        expfmt = format(exponent,".2f").rstrip("0").rstrip(".")
        return f'{dimension_name}**{expfmt}'
    else:
        return ''

def _fmt_dimensions(
    exponents: Iterable[float], 
    dimension_names: Iterable[str]
    ) -> str:
    if u.dimensionless(exponents):
        return '1'
    else:
        fmts = [_fmt_dimension(e,n) for e,n in zip(exponents, dimension_names)]
        fmts = [f for f in fmts if f != '']
        return '*'.join(fmts)

class DimensionalSystem:
    def __init__(self, base_dims: Union[Iterable[str], int]):
        if isinstance(base_dims, int):
            base_dims = string.ascii_uppercase[:base_dims]
        self.base_dims = list(base_dims)
        # Export vars as attributes of this instance
        for n,q in zip(self.base_dims, self.base_quantities()):
            setattr(self, n, q)

    def __len__(self) -> int:
        return len(self.base_dims)

    def q(self, exponents: Optional[Union[Iterable[float], str]]=None) -> Q:
        if exponents is None:
            return Q(self, np.zeros(len(self)))
        elif isinstance(exponents, str):
            return Q(
                self, 
                u.basis_vec(self.base_dims.index(exponents), len(self))
            )
        else:
            return Q(self, exponents)

    def base_quantities(self) -> Iterable[Q]:
        for b in self.base_dims:
            yield self.q(b)

    def __call__(
            self, 
            exponents: Optional[Union[Iterable[float], str]]=None
        ) -> Q:
        return self.q(exponents)

    def __eq__(self, other) -> bool:
        if isinstance(other, DimensionalSystem):
            return self.base_dims == other.base_dims
        return False

    def qs_from_dm(self, dm: np.ndarray) -> List[Q]:
        dmt = np.atleast_2d(dm).T
        return [self.q(e) for e in dmt]

    @property
    def Unity(self):
        return self()

class Q:
    def __init__(self, dimsys: DimensionalSystem, exponents: Iterable[float]):
        e = np.asarray(exponents).astype(float)
        assert len(e) == len(dimsys)
        self.e = e
        self.dimsys = dimsys

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({str(self)})'   

    def __str__(self) -> str:        
        return _fmt_dimensions(self.e, self.dimsys.base_dims)

    def __array__(self) -> Iterable[float]:
        return self.e  

    def __len__(self) -> int:
        return self.e.shape[0]

    def __iter__(self) -> Iterator[float]:
        return iter(self.e)

    def __getitem__(self, idx) -> float:
        return self.e[idx]

    def __eq__(self, other) -> bool:
        if isinstance(other, Q):
            return (
                self.dimsys == other.dimsys and
                np.allclose(self.e, other.e)
            )
        return False
    
    def __pow__(self, exponent) -> Q:
        return self.q(self.e*exponent)

    def __mul__(self, other: Q) -> Q:
        other = np.asarray(other)
        return self.q(self.e + other)

    def __rmul__(self, other: Q) -> Q:
        return self.__mul__(other)

    def __truediv__(self, other) -> Q:    
        other = np.asarray(other)
        return self.q(self.e + (other*-1))

    def __rtruediv__(self, other) -> Q:    
        return self.__truediv__(other)

    def q(self, e: Optional[Union[Iterable[float], str]] = None) -> Q:
        return self.dimsys.q(e)

    @property
    def is_dimensionless(self) -> bool:
        return np.allclose(self.e, 0.)
        
