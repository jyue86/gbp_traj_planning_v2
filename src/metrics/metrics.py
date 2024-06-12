import jax.numpy as jnp

def log_dimensionless_jerk(waypoints: jnp.ndarray) -> float:
    return jnp.sum(jnp.linalg.norm(jnp.diff(waypoints, axis=0), axis=1))

def total_dist_travelled(velocities: jnp.ndarray, duration: float) -> float:
    max_velocity = jnp.max(jnp.linalg.norm(velocities, axis=1))
    scale = jnp.pow(duration, 3)/jnp.pow(max_velocity, 2)
    jerk = jnp.pow(jnp.linalg.norm(jnp.diff(velocities, n=2, axis=0)/jnp.pow(1/duration,2)), 2)
    return -jnp.log((scale*jerk)/duration)