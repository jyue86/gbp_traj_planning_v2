import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class Gaussian:
    info: jnp.array
    precision: jnp.array

    def __add__(self, other_gaussian: "Gaussian"):
        return Gaussian(self.info + other_gaussian.info, self.precision + other_gaussian.precision)
    
    @staticmethod
    def identity():
        return Gaussian(jnp.zeros(4), jnp.eye(4))