from __future__ import annotations 
import numpy as np
import string
from typing import Union, Iterable, Optional
from . import utils as u

def return_quantity(func):
    '''Decorator to return wrap result in corresponding Quantity class.'''
    def wrapper(self, *args, **kwargs):
        cls = self.__class__
        return cls(func(self, *args, **kwargs))
    return wrapper

def _fmt_dimension(exponent, dimension_name):
    if np.allclose(exponent, 1): # misuse of allclose for scalars
        return dimension_name
    elif not np.allclose(exponent, 0):
        expfmt = format(exponent,".2f").rstrip("0").rstrip(".")
        return f'{dimension_name}**{expfmt}'
    else:
        return ''

def _fmt_dimensions(exponents, dimension_names):
    if u.dimensionless(exponents):
        return '1'
    else:
        fmts = [_fmt_dimension(e,n) for e,n in zip(exponents, dimension_names)]
        fmts = [f for f in fmts if f != '']
        return '*'.join(fmts)

class Quantity:
    def __init__(self, exponents):        
        self.e = np.asarray(exponents).astype(float)

    def __repr__(self):
        cls = self.__class__
        return f'<{cls.__name__} {str(self)}>'       

    def __str__(self):
        cls = self.__class__
        return _fmt_dimensions(self.e, cls.basedims())

    @return_quantity
    def __pow__(self, exponent):
        return self.e*exponent

    @return_quantity
    def __mul__(self, other):
        other = np.asarray(other)
        return self.e + other

    @return_quantity
    def __rmul__(self, other):
        other = np.asarray(other)
        return self.e + other

    @return_quantity
    def __truediv__(self, other):    
        other = np.asarray(other)
        return self.e + (other*-1)

    @return_quantity
    def __rtruediv__(self, other):    
        other = np.asarray(other)
        return self.e + (other*-1)

    def __array__(self):
        return self.e

    def __iter__(self):
        return iter(self.e)

    def __len__(self):
        return len(self.e)

    def __getitem__(self, idx):
        return self.e[idx]

    @property
    def exponents(self):
        return self.e

    @property
    def dimensionless(self):
        return np.allclose(self.e, 0.)

    @property
    def shape(self):
        return self.e.shape   

    def as_quantity(self, exponents):
        cls = self.__class__
        return cls(exponents)

    @classmethod
    def basedims(cls):
        if hasattr(cls, '__BASEDIMS__'):
            return cls.__BASEDIMS__
        else:
            return string.ascii_uppercase

    @classmethod
    def unity(cls, ndims=None):
        if not hasattr(cls, '__BASEDIMS__'):
            if ndims is None:
                raise ValueError('Unity is undefined for unknown number of dims.')
            return cls(np.zeros(ndims))
        else:
            return cls(np.zeros(len(cls.__BASEDIMS__)))

    @classmethod
    def basevars(cls):
        if not hasattr(cls, '__BASEDIMS__'):
            raise ValueError('Unknown dimensions.')
        N = len(cls.__BASEDIMS__)
        return [cls(u.basis_vec(i,N)) for i in range(N)]        

    @classmethod
    def create_type(cls, name, basedims):
        def initializer(self, exponents=None):
            if exponents is None:
                cls = self.__class__
                exponents = cls.unity()
            Quantity.__init__(self, exponents)

        kls = type(
            name, 
            (Quantity,), 
            {'__BASEDIMS__':basedims, '__init__':initializer}
        )
        return kls


class DimensionalSystem:
    def __init__(self, base_dims: Union[Iterable[str], int]):
        if isinstance(base_dims, int):
            base_dims = string.ascii_uppercase[:base_dims]
        self.base_dims = list(base_dims)

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

    def __eq__(self, other):
        if isinstance(other, DimensionalSystem):
            return self.base_dims == other.base_dims
        return False

    @property
    def unity(self):
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

    def __iter__(self):
        return iter(self.e)

    def __getitem__(self, idx):
        return self.e[idx]

    def __eq__(self, other):
        if isinstance(other, Q):
            return (
                len(self) == len(other) and
                self.e == other.e and
                self.dimsys == other.dimsys
            )
        return False
    
    def __pow__(self, exponent) -> Q:
        return self.q(self.e*exponent)

    def __mul__(self, other: Q) -> Q:
        other = np.asarray(other)
        return self.q(self.e + other)

    def __rmul__(self, other: Q) -> Q:
        return self.__mul__(other)

    def __truediv__(self, other):    
        other = np.asarray(other)
        return self.q(self.e + (other*-1))

    def __rtruediv__(self, other):    
        return self.__truediv__(other)

    def q(self, e: Optional[Union[Iterable[float], str]] = None) -> Q:
        return self.dimsys.q(e)

    @property
    def is_dimensionless(self):
        return np.allclose(self.e, 0.)
        
