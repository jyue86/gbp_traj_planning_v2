import jax.numpy as jnp

def total_dist_travelled(waypoints: jnp.ndarray) -> float:
    return jnp.sum(jnp.linalg.norm(jnp.diff(waypoints, axis=0), axis=1))

def log_dimensionless_jerk(waypoints: jnp.ndarray, duration: float) -> float:
    max_velocity = jnp.max(jnp.linalg.norm(jnp.diff(waypoints, axis=0)/duration, axis=1))
    scale = jnp.pow(duration, 3)/jnp.pow(max_velocity, 2)
    jerk = jnp.pow(jnp.linalg.norm(jnp.diff(waypoints, n=3, axis=0)/jnp.pow(1/duration,3)), 2)
    return -jnp.log((scale*jerk)/duration)