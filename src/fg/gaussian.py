"""
Marginalization function borrowed from
https://github.com/NikuKikai/Gaussian-Belief-Propagation-on-Planning/blob/main/src/fg/gaussian.py
"""


from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class Gaussian:
    info: jnp.ndarray
    precision: jnp.ndarray
    _dims: Tuple 
 
    @property
    def mean(self) -> jnp.ndarray:
        return jnp.linalg.inv(self.precision) @ self.info
    
    @property
    def covariance(self) -> jnp.ndarray:
        return jnp.linalg.inv(self.precision)
    
    @staticmethod
    def identity() -> jnp.ndarray:
        return Gaussian(jnp.zeros(4), jnp.eye(4))
    
    def __getitem__(self, index) -> "Gaussian":
        return Gaussian(self.info[index], self.precision[index])

    def __mul__(self, other: 'Gaussian') -> 'Gaussian':
        if other is None:
            return self.copy()

        # Merge dims
        dims = list(self._dims)
        for d in other._dims:
            if d not in dims:
                dims.append(d)
        
        # Extend self matrix
        prec_self = jnp.zeros((len(dims), len(dims)))
        info_self = jnp.zeros((len(dims), 1))
        idxs_self = jnp.array([dims.index(d) for d in self._dims]) # here, need to fix this
        prec_self = prec_self.at[jnp.ix_(idxs_self, idxs_self)].set(self.precision)
        info_self = info_self.at[jnp.ix_(idxs_self,jnp.array([0]))].set(self.info.reshape(-1,1))

        # Extend other matrix
        prec_other = jnp.zeros((len(dims), len(dims)))
        info_other = jnp.zeros((len(dims), 1))
        idxs_other = jnp.array([dims.index(d) for d in other._dims]) # here, need to fix this
        prec_other = prec_other.at[jnp.ix_(idxs_other, idxs_other)].set(other.precision)
        info_other = info_other.at[jnp.ix_(idxs_other, jnp.array([0]))].set(other.info.reshape(-1,1))
        # Add
        prec = prec_other + prec_self
        info = info_other + info_self
        return Gaussian(info, prec, dims)

    def __imul__(self, other: 'Gaussian') -> 'Gaussian':
        return self.__mul__(other)
    
    def marginalize(self, dims: Tuple) -> "Gaussian":
        info, prec = self.info, self.precision
        axis_a = [idx for idx, d in enumerate(self._dims) if d not in dims]
        axis_b = [idx for idx, d in enumerate(self._dims) if d in dims]

        def axis_a_fn(kp, v):
            if v not in dims:
                return kp[0].idx
            else:
                return -1
            
        def axis_b_fn(kp, v):
            if v in dims:
                return kp[0].idx
            else:
                return -1
            
        axis_a = jnp.array(jax.tree_util.tree_map_with_path(axis_a_fn, self._dims))
        axis_b = jnp.array(jax.tree_util.tree_map_with_path(axis_b_fn, self._dims))
        axis_a = axis_a[jnp.where(axis_a != -1)]
        axis_b = axis_b[jnp.where(axis_b != -1)]

        info_a = info[jnp.ix_(axis_a, jnp.array([0]))]
        prec_aa = prec[jnp.ix_(axis_a, axis_a)]
        info_b = info[jnp.ix_(axis_b, jnp.array([0]))]
        prec_ab = prec[jnp.ix_(axis_a, axis_b)]
        prec_ba = prec[jnp.ix_(axis_b, axis_a)]
        prec_bb = prec[jnp.ix_(axis_b, axis_b)]

        prec_bb_inv = jnp.linalg.inv(prec_bb)
        info_ = info_a - prec_ab @ prec_bb_inv @ info_b
        prec_ = prec_aa - prec_ab @ prec_bb_inv @ prec_ba

        dims = tuple(i for i in self._dims if i not in dims)
        return Gaussian(info_, prec_, dims)