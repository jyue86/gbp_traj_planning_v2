from abc import abstractmethod
import jax
import jax.numpy as jnp

# from env.obstacle import Obstacle
from .gaussian import Gaussian

N_STATES = 4
POSE_NOISE = 1e-15
DYNAMICS_NOISE = 0.005
OBSTACLE_NOISE = 0.005

class Factor(Gaussian):
    def __init__(self, state: jnp.array, state_precision: jnp.array, linear:bool=True) -> None:
        self.linear = linear 
        super().__init__(self._calc_info(state, state_precision), self._calc_precision(state, state_precision))
    
    @abstractmethod
    def _calc_measurement(self, state: jnp.array) -> jnp.array:
        pass

    def _calc_info(self, state: jnp.array, precision: jnp.array) -> jnp.array:
        X = state
        if self.linear:
            eta = precision @ (jnp.zeros(N_STATES) - self._calc_measurement(state))
        else:
            J = jax.jacfwd(self._calc_measurement)(state)
            eta = (J.T @ precision) @ (J @ X + jnp.zeros((X.shape[0], 1)) - self.calc_measurement(state))
        return eta
    
    def _calc_precision(self, state: jnp.array, precision: jnp.array) -> None:
        if self.linear:
            return precision
        else:
            J = jax.jacfwd(self._calc_measurement)(state)
            return J.T @ precision @ J

class PoseFactor(Factor):
    def __init__(self, state: jnp.array) -> None:
        precision = jnp.pow(POSE_NOISE, -2) * jnp.eye(N_STATES)
        super(PoseFactor, self).__init__(state, precision)
    
    def _calc_measurement(self, state) -> jnp.array:
        return state

class DynamicsFactor(Factor):
    def __init__(
        self,
        state: jnp.array,
        delta_t: float
    ) -> None:
        self.delta_t = delta_t
        process_covariance = DYNAMICS_NOISE * jnp.eye(N_STATES//2)
        top_half = jnp.hstack(
            (
                self.delta_t**3 * process_covariance / 3,
                self.delta_t**2 * process_covariance / 2,
            )
        )
        bottom_half = jnp.hstack(
            (
                self.delta_t**2 * process_covariance / 2,
                self.delta_t * process_covariance,
            )
        )
        precision = jnp.vstack((top_half, bottom_half))
        precision = jnp.linalg.inv(precision)

        self.state_transition = jnp.eye(4)
        self.state_transition = self.state_transition.at[0:2,2:].set(jnp.eye(2) * self.delta_t)


        super(DynamicsFactor, self).__init__(state, precision)

    def _calc_measurement(self, state: jnp.array) -> jnp.array:
        prev_state = state[0:4]
        current_state = state[4:]
        return self.state_transition @ prev_state - current_state