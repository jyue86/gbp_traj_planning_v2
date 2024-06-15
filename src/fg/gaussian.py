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
    
    def __call__(self, x: jnp.ndarray) -> float:
        return 1/2 * x.T @ self.precision @ x - self.info.T @ x

    def __getitem__(self, index) -> "Gaussian":
        return Gaussian(self.info[index], self.precision[index], self.dims[index])

    def __mul__(self, other: 'Gaussian') -> 'Gaussian':
        if other is None:
            return self.copy()
        
        dims = self.dims.copy()
        other_dims = other.dims.copy()
        other_unique_val = jnp.unique(other_dims, size=other_dims.shape[0]//4)[0]
        
        if dims.shape[0] == other_dims.shape[0] and dims[0] == other_dims[0]:
            idxs_self = idxs_other = jnp.arange(len(dims))
        elif dims.shape[0] == 8:
            idxs_self = jnp.arange(len(dims), dtype=int)
            idxs_other = jnp.where(dims == other_unique_val, size=4)[0]
        else:
            idxs_self = jnp.arange(len(dims), dtype=int)
            idxs_other = jnp.arange(len(dims), len(dims) + len(other_dims)) # jax.lax.select((dims == other_dims).sum() == dims.shape[0], jnp.arange(len(dims)), jnp.arange(len(dims), len(dims) + len(other_dims)))
            dims = jnp.concatenate((dims, other_dims))

            # if idxs_other[-1] != idxs_self[-1]:
            #     dims = jnp.concatenate((dims, other_dims))
            
        
        # Extend self matrix
        prec_self = jnp.zeros((len(dims), len(dims)))
        info_self = jnp.zeros((len(dims), 1))
        
        prec_self = prec_self.at[jnp.ix_(idxs_self, idxs_self)].set(self.precision)
        info_self = info_self.at[jnp.ix_(idxs_self, jnp.zeros(1, dtype=int))].set(self.info.reshape(len(self.dims), 1))
        # Extend other matrix
        prec_other = jnp.zeros((len(dims), len(dims)))
        info_other = jnp.zeros((len(dims), 1))
        
        prec_other = prec_other.at[jnp.ix_(idxs_other, idxs_other)].set(other.precision)
        info_other = info_other.at[jnp.ix_(idxs_other, jnp.zeros((1), dtype=int))].set(other.info.reshape(len(other.dims), 1))
        # Add
        prec = prec_other + prec_self
        info = info_other + info_self
        return Gaussian(info, prec, dims.astype(float))
    
    def mul(self, other: 'Gaussian', combine_dims: bool) -> 'Gaussian':
        if other is None:
            return self.copy()
        
        dims = self.dims.copy()
        other_dims = other.dims.copy()
        other_unique_val = jnp.unique(other_dims, size=other_dims.shape[0]//4)[0]
        
        if dims.shape[0] == 8:
            idxs_self = jnp.arange(len(dims), dtype=int)
            idxs_other = jnp.where(dims == other_unique_val, size=4)[0]
        else:
            idxs_self = jnp.arange(len(dims), dtype=int)
            idxs_other = jax.lax.select((dims == other_dims).sum() == dims.shape[0], jnp.arange(len(dims)), jnp.arange(len(dims), len(dims) + len(other_dims)))
            
            if combine_dims:
                dims = jnp.concatenate((dims, other_dims))

        # Extend self matrix
        prec_self = jnp.zeros((len(dims), len(dims)))
        info_self = jnp.zeros((len(dims), 1))
        
        prec_self = prec_self.at[jnp.ix_(idxs_self, idxs_self)].set(self.precision)
        info_self = info_self.at[jnp.ix_(idxs_self, jnp.zeros(1, dtype=int))].set(self.info.reshape(len(self.dims), 1))
        # Extend other matrix
        prec_other = jnp.zeros((len(dims), len(dims)))
        info_other = jnp.zeros((len(dims), 1))
        
        prec_other = prec_other.at[jnp.ix_(idxs_other, idxs_other)].set(other.precision)
        info_other = info_other.at[jnp.ix_(idxs_other, jnp.zeros((1), dtype=int))].set(other.info.reshape(len(other.dims), 1))
        # Add
        prec = prec_other + prec_self
        info = info_other + info_self
        return Gaussian(info.squeeze(), prec, dims.astype(float))
    
    def marginalize(self, dims_to_remove) -> "Gaussian":
        info, prec = self.info.reshape(-1, 1), self.precision
        
        axis_a = jnp.where(self.dims != dims_to_remove[0], size=4)[0]
        axis_b = jax.lax.select(axis_a[0] == 0, jnp.arange(4, 8), jnp.arange(4))

        info_a = info[jnp.ix_(axis_a, jnp.zeros((1,), dtype=int))]
        prec_aa = prec[jnp.ix_(axis_a, axis_a)]
        info_b = info[jnp.ix_(axis_b, jnp.zeros((1, ), dtype=int))]
        prec_ab = prec[jnp.ix_(axis_a, axis_b)]
        prec_ba = prec[jnp.ix_(axis_b, axis_a)]
        prec_bb = prec[jnp.ix_(axis_b, axis_b)]

        # diag_values = jnp.diag(prec_bb)
        # prec_bb = jnp.fill_diagonal(prec_bb, jnp.where(diag_values == 0, diag_values[0], diag_values), inplace=False)

        prec_bb_inv = jnp.linalg.inv(prec_bb)
        info_ = (info_a - prec_ab @ prec_bb_inv @ info_b).squeeze()
        prec_ = (prec_aa - prec_ab @ prec_bb_inv @ prec_ba).squeeze()

        new_dims = jnp.full(4, self.dims[axis_a[0]])

        info = jax.lax.select(jnp.logical_not(jnp.logical_and(jnp.all(jnp.isfinite(info_)), jnp.all(jnp.isfinite(prec_)))), jnp.zeros_like(info_), info_)
        prec = jax.lax.select(jnp.logical_not(jnp.logical_and(jnp.all(jnp.isfinite(info_)), jnp.all(jnp.isfinite(prec_)))), jnp.zeros_like(prec_), prec_)

        return Gaussian(info, prec, new_dims)