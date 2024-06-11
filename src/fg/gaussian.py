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

        dims = self.dims.copy()
        other_dims = other.dims.copy()
        unique_dim_values = jnp.unique(self.dims, size=self.dims.shape[0]//4)
        unique_other_dim_values = jnp.unique(other_dims, size=other_dims.shape[0]//4)

        def check_missing_index(carry, x):
            mask = jnp.where(carry == x, 1.0, 0.0)
            result = jax.lax.select(mask.sum() == 0, jnp.full((4,), True), jnp.full((4,), False))
            return carry, result 

        def get_indices(carry, x):
            mask = jnp.where(carry == x, x, 0.0)
            return carry, jnp.array([0.,1.,2.,3.]) + jnp.nonzero(mask, size=1)[0]

        _, missing_axes_values = jax.lax.scan(check_missing_index, dims, unique_other_dim_values)
        missing_axes_values = other_dims[missing_axes_values.flatten()] 
        unique_missing_axes_values = jnp.unique(missing_axes_values, size=missing_axes_values.shape[0]//4)
        _, missing_axes = jax.lax.scan(get_indices, other_dims, unique_missing_axes_values)
        dims = jnp.concat((dims, other_dims[missing_axes.flatten().astype(int)])) # nonzero usage will break vmap
        # jax.debug.print("dims: {}", dims)
        
        # Extend self matrix
        prec_self = jnp.zeros((len(dims), len(dims)))
        info_self = jnp.zeros((len(dims), 1))
        _, idxs_self = jax.lax.scan(get_indices, dims, unique_dim_values)
        idxs_self = idxs_self.flatten().astype(int)
        # jax.debug.print("idxs_self: {}", idxs_self)
        prec_self = prec_self.at[jnp.ix_(idxs_self, idxs_self)].set(self.precision)
        info_self = info_self.at[jnp.ix_(idxs_self,jnp.array([0]))].set(self.info.reshape(-1,1))

        # Extend other matrix
        prec_other = jnp.zeros((len(dims), len(dims)))
        info_other = jnp.zeros((len(dims), 1))
        _, idxs_other = jax.lax.scan(get_indices, dims, unique_other_dim_values)
        idxs_other = idxs_other.flatten().astype(int)
        # jax.debug.print("idxs_other: {}", idxs_other)
        prec_other = prec_other.at[jnp.ix_(idxs_other, idxs_other)].set(other.precision)
        info_other = info_other.at[jnp.ix_(idxs_other, jnp.array([0]))].set(other.info.reshape(-1,1))

        # Add
        prec = prec_other + prec_self
        info = (info_other + info_self).squeeze(-1)
        return Gaussian(info, prec, dims)

    def __imul__(self, other: 'Gaussian') -> 'Gaussian':
        return self.__mul__(other)
    
    def marginalize(self, marginalized_dims: jnp.ndarray) -> "Gaussian":
        info, prec = self.info, self.precision
        info = info.reshape(-1,1)
        unique_dim_values = jnp.unique(marginalized_dims, size=marginalized_dims.shape[0]//4)
        
        def find_axis_a_values(carry, x):
            mask = jnp.where(carry == x, False, True)
            return carry, mask

        def find_axis_b(carry, x):
            mask = jnp.where(carry == x, True, False)
            return carry, mask
        
        def get_indices(carry, x):
            mask = jnp.where(carry == x, 1.0, 0.0)
            return carry, jnp.array([0., 1., 2., 3.]) + jnp.nonzero(mask, size=1)[0]
        _, axis_a_values = jax.lax.scan(find_axis_a_values, self.dims, unique_dim_values)
        axis_a_values = self.dims[axis_a_values.flatten()]
        unique_axis_a_values = jnp.unique(axis_a_values, size=axis_a_values.shape[0]//4)
        _, axis_a = jax.lax.scan(get_indices, self.dims, unique_axis_a_values)
        axis_a = axis_a.flatten().astype(int)

        _, axis_b_values = jax.lax.scan(find_axis_b, self.dims, unique_dim_values)
        axis_b_values = self.dims[axis_b_values.flatten()]
        unique_axis_b_values = jnp.unique(axis_b_values, size=axis_b_values.shape[0]//4)
        _, axis_b = jax.lax.scan(get_indices, self.dims, unique_axis_b_values)
        axis_b = axis_b.flatten().astype(int)

        # jax.debug.print("axis a: {}", axis_a)
        # jax.debug.print("axis b: {}", axis_b)

        info_a = info[jnp.ix_(axis_a, jnp.array([0]))]
        prec_aa = prec[jnp.ix_(axis_a, axis_a)]
        info_b = info[jnp.ix_(axis_b, jnp.array([0]))]
        prec_ab = prec[jnp.ix_(axis_a, axis_b)]
        prec_ba = prec[jnp.ix_(axis_b, axis_a)]
        prec_bb = prec[jnp.ix_(axis_b, axis_b)]

        prec_bb_inv = jnp.linalg.inv(prec_bb)
        info_ = info_a - prec_ab @ prec_bb_inv @ info_b
        prec_ = prec_aa - prec_ab @ prec_bb_inv @ prec_ba

        return Gaussian(info_.squeeze(-1), prec_, axis_a_values)