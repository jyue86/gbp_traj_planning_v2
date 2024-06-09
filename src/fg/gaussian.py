"""
Marginalization function borrowed from
https://github.com/NikuKikai/Gaussian-Belief-Propagation-on-Planning/blob/main/src/fg/gaussian.py
"""


from typing import List

import jax
import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class Gaussian:
    info: jnp.array
    precision: jnp.array

    def __add__(self, other_gaussian: "Gaussian") -> "Gaussian":
        return Gaussian(self.info + other_gaussian.info, self.precision + other_gaussian.precision)
    
    @property
    def mean(self) -> jnp.array:
        return jnp.linalg.inv(self.precision) @ self.info
    
    @property
    def covariance(self) -> jnp.array:
        return jnp.linalg.inv(self.precision)
    
    @staticmethod
    def identity() -> jnp.array:
        return Gaussian(jnp.zeros(4), jnp.eye(4))
    
    # TODO: idk what to do about dims rn
    def marginalize(self, dims: List) -> "Gaussian":
        info, prec = self._info, self._prec
        axis_a = [idx for idx, d in enumerate(self._dims) if d not in dims]
        axis_b = [idx for idx, d in enumerate(self._dims) if d in dims]
        info_a = info[jnp.ix_(axis_a, [0])]
        prec_aa = prec[jnp.ix_(axis_a, axis_a)]
        info_b = info[jnp.ix_(axis_b, [0])]
        prec_ab = prec[jnp.ix_(axis_a, axis_b)]
        prec_ba = prec[jnp.ix_(axis_b, axis_a)]
        prec_bb = prec[jnp.ix_(axis_b, axis_b)]

        prec_bb_inv = jnp.linalg.inv(prec_bb)
        info_ = info_a - prec_ab @ prec_bb_inv @ info_b
        prec_ = prec_aa - prec_ab @ prec_bb_inv @ prec_ba

        new_dims = tuple(d for d in self._dims if d not in dims)
        return Gaussian.from_info(new_dims, info_, prec_)