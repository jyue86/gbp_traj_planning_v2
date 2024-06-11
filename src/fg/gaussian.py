"""
Marginalization function borrowed from
https://github.com/NikuKikai/Gaussian-Belief-Propagation-on-Planning/blob/main/src/fg/gaussian.py
"""


import jax
import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class Gaussian:
    info: jnp.ndarray
    precision: jnp.ndarray
    dims: jnp.ndarray 

    @property
    def shape(self):
        return {
            "info": self.info.shape,
            "precision": self.precision.shape,
            "dims": self.dims.shape
        }
 
    @property
    def mean(self) -> jnp.ndarray:
        return jnp.linalg.inv(self.precision) @ self.info
    
    @property
    def covariance(self) -> jnp.ndarray:
        return jnp.linalg.inv(self.precision)
    
    @staticmethod
    def identity(variable: int) -> jnp.ndarray:
        dims = jnp.array([variable, variable, variable, variable])
        return Gaussian(jnp.zeros(4), jnp.eye(4), dims)
    
    def concatenate(self, other_gaussian: "Gaussian") -> "Gaussian":
        return Gaussian(
            jnp.concatenate(self.info, other_gaussian.info),
            jnp.concatenate(self.precision, other_gaussian.precision),
            jnp.concatenate(self.dims, other_gaussian.dims)
        )
    
    def __getitem__(self, index) -> "Gaussian":
        return Gaussian(self.info[index], self.precision[index], self.dims[index])

    def __mul__(self, other: 'Gaussian') -> 'Gaussian':
        if other is None:
            return self.copy()

        # Merge dims
        # dims = [i for i in self.dims]
        # for d in other.dims:
        #     if d not in dims:
        #         dims.append(d)
        dims = self.dims
        if other.dims.shape != dims.shape or jnp.sum(dims == other.dims) != dims.shape[0]:
            dims = jnp.concat((dims, other.dims))       #         dims.append(d)
        
        # Extend self matrix
        prec_self = jnp.zeros((len(dims), len(dims)))
        info_self = jnp.zeros((len(dims), 1))
        # idxs_self = jnp.array([dims.index(d) for d in self.dims]) # here, need to fix this
        # jax.debug.print("{}, {}", self.dims, self.dims.shape)
        # idxs_self = jnp.concat([jnp.where(dims == d)[0].flatten() for d in jnp.unique(self.dims, size=2)])
        _, indices = jnp.unique(other.dims, return_index=True, size=2)
        idxs_self = []
        for i, d in enumerate(indices):
            idxs_self.append(jnp.array([0., 1., 2., 3.]) + i)
        idxs_self = jnp.concatenate(idxs_self) 
        prec_self = prec_self.at[jnp.ix_(idxs_self, idxs_self)].set(self.precision)
        info_self = info_self.at[jnp.ix_(idxs_self,jnp.array([0]))].set(self.info.reshape(-1,1))

        # Extend other matrix
        prec_other = jnp.zeros((len(dims), len(dims)))
        info_other = jnp.zeros((len(dims), 1))
        # idxs_other = jnp.array([dims.index(d) for d in other.dims]) # here, need to fix this
        _, indices = jnp.unique(other.dims, return_index=True, size=2)
        idxs_other = []
        for i, d in enumerate(indices):
            idxs_other.append(jnp.array([0., 1., 2., 3.]) + i)
        idxs_other = jnp.concatenate(idxs_other) 
        prec_other = prec_other.at[jnp.ix_(idxs_other, idxs_other)].set(other.precision)
        info_other = info_other.at[jnp.ix_(idxs_other, jnp.array([0]))].set(other.info.reshape(-1,1))
        # Add
        prec = prec_other + prec_self
        info = (info_other + info_self).squeeze(-1)
        return Gaussian(info, prec, dims)

    def __imul__(self, other: 'Gaussian') -> 'Gaussian':
        return self.__mul__(other)
    
    def marginalize(self, dims: jnp.ndarray) -> "Gaussian":
        info, prec = self.info, self.precision
        info = info.reshape(-1,1)
        axis_a = jnp.array([idx for idx, d in enumerate(self.dims) if d not in dims])
        axis_b = jnp.array([idx for idx, d in enumerate(self.dims) if d in dims])

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
            
        # axis_a = jnp.array(jax.tree_util.tree_map_with_path(axis_a_fn, self.dims))
        # axis_b = jnp.array(jax.tree_util.tree_map_with_path(axis_b_fn, self.dims))
        # axis_a = axis_a[jnp.where(axis_a != -1)]
        # axis_b = axis_b[jnp.where(axis_b != -1)]

        info_a = info[jnp.ix_(axis_a, jnp.array([0]))]
        prec_aa = prec[jnp.ix_(axis_a, axis_a)]
        info_b = info[jnp.ix_(axis_b, jnp.array([0]))]
        prec_ab = prec[jnp.ix_(axis_a, axis_b)]
        prec_ba = prec[jnp.ix_(axis_b, axis_a)]
        prec_bb = prec[jnp.ix_(axis_b, axis_b)]

        prec_bb_inv = jnp.linalg.inv(prec_bb)
        info_ = info_a - prec_ab @ prec_bb_inv @ info_b
        prec_ = prec_aa - prec_ab @ prec_bb_inv @ prec_ba

        dims = tuple(i for i in self.dims if i not in dims)
        return Gaussian(info_.squeeze(-1), prec_, dims)