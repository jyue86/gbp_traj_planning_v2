import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class Obstacle:
    pos: jnp.array # Nx2 for N obstacles, (x,y) position