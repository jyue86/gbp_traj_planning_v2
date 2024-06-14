from abc import abstractmethod
import jax
import jax.numpy as jnp

# from env.obstacle import Obstacle
from .gaussian import Gaussian

N_STATES = 4
POSE_NOISE = 1e-15
DYNAMICS_NOISE = 0.005
OBSTACLE_NOISE = 0.005
INTER_ROBOT_NOISE = 0.005


class Factor:
    def __init__(
        self, state: jnp.array, state_precision: jnp.ndarray, dims: jnp.array, linear: bool = True
    ) -> None:
        self._state = state
        self._state_precision = state_precision
        self._linear = linear
        self._dims = dims

    def calculate_likelihood(self) -> Gaussian:
        return Gaussian(
            self._calc_info(self._state, self._state_precision),
            self._calc_precision(self._state, self._state_precision),
            self._dims
        )

    @abstractmethod
    def _calc_measurement(self, state: jnp.ndarray) -> jnp.ndarray:
        pass

    def _calc_info(self, state: jnp.ndarray, precision: jnp.ndarray) -> jnp.ndarray:
        X = state
        if self._linear:
            eta = precision @ (jnp.zeros(N_STATES) - self._calc_measurement(state))
        else:
            J = jax.jacfwd(self._calc_measurement)(state)
            # jax.debug.breakpoint()
            eta = (J.T @ precision) @ (
                (J @ X.reshape((-1,1))) + 0 - self._calc_measurement(state).reshape((-1,1))
            )
            # jax.debug.breakpoint()
        return eta.squeeze()

    def _calc_precision(self, state: jnp.ndarray, precision: jnp.ndarray) -> jnp.ndarray:
        if self._linear:
            return precision
        else:
            J = jax.jacfwd(self._calc_measurement)(state)
            return J.T @ precision @ J


class PoseFactor(Factor):
    def __init__(self, state: jnp.ndarray, dims: jnp.ndarray) -> None:
        precision = jnp.pow(POSE_NOISE, -2) * jnp.eye(N_STATES)
        super(PoseFactor, self).__init__(state, precision, dims)

    def _calc_measurement(self, state: jnp.ndarray) -> jnp.ndarray:
        return state


class DynamicsFactor(Factor):
    def __init__(self, state: jnp.ndarray, delta_t: float, dims: jnp.ndarray) -> None:
        self._delta_t = delta_t
        process_covariance = DYNAMICS_NOISE * jnp.eye(N_STATES // 2)
        top_half = jnp.hstack(
            (
                self._delta_t**3 * process_covariance / 3,
                self._delta_t**2 * process_covariance / 2,
            )
        )
        bottom_half = jnp.hstack(
            (
                self._delta_t**2 * process_covariance / 2,
                self._delta_t * process_covariance,
            )
        )
        precision = jnp.vstack((top_half, bottom_half))
        precision = jnp.linalg.inv(precision)
        # precision = jnp.diag(jnp.array([10, 10, 20, 20]))

        self._state_transition = jnp.eye(4)
        self._state_transition = self._state_transition.at[0:2, 2:].set(
            jnp.eye(2) * self._delta_t
        )

        super(DynamicsFactor, self).__init__(state, precision, dims, linear=False)

    def _calc_measurement(self, state: jnp.ndarray) -> jnp.ndarray:
        prev_state = state[0:4]
        current_state = state[4:]
        diff = self._state_transition @ prev_state - current_state
        return diff
    
    # def _calc_info(self, state: jnp.ndarray, precision: jnp.ndarray) -> jnp.ndarray:
    #     X = state
    #     J = jnp.array([
    #         [1, 0, self._delta_t, 0, -1, 0, 0, 0],  # h(x)[0] = dx = x(k) + vx(k) * dt - x(k+1)
    #         [0, 1, 0, self._delta_t, 0, -1, 0, 0],  # h(x)[1] = dy = y(k) + vy(k) * dt - y(k+1)
    #         [0, 0, 1, 0, 0, 0, -1, 0],  # h(x)[2] = dvx = vx(k) - vx(k+1)
    #         [0, 0, 0, 1, 0, 0, 0, -1],  # h(x)[3] = dvy = vy(k) - vy(k+1)
    #     ])  # [4, 8]
    #     h = self._calc_measurement(state)
    #     first_part = (J.T @ precision)
    #     second_part = (J @ X.reshape((-1,1)) - h.reshape((-1,1)))
    #     eta = first_part @ second_part 
    #     return eta.squeeze()
    
class InterRobotFactor(Factor):
    def __init__(
        self,
        state: jnp.ndarray,
        agent_radius: float,
        critical_distance: float,
        t: jnp.ndarray, #ndarray just to hold time,
        dims: jnp.ndarray,
    ) -> None:
        self._critical_distance = critical_distance
        self._agent_radius = agent_radius
        precision = jnp.pow(t * INTER_ROBOT_NOISE, -2) * jnp.eye(N_STATES)
        super(InterRobotFactor, self).__init__(state, precision, dims, False)

    def _calc_measurement(self, state: jnp.ndarray):
        current_state = state[0:4]
        other_state = state[4:]
        dist = self._calc_dist(current_state, other_state)
        # jax.debug.print("dist: {}", dist)
        measurement = jax.lax.select(
            dist < self._critical_distance, jnp.full((4,), 1.0 - dist / self._critical_distance), jnp.zeros((4,)) 
        )
        def breakpoint_if_less(dist):
            cond = dist < self._critical_distance
            def true_fn(x):
                jax.debug.breakpoint()
            def false_fn(x):
                pass
            jax.lax.cond(cond, true_fn, false_fn, dist)
        breakpoint_if_less(dist)
        return measurement
    
    def _calc_dist(self, state: jnp.array, other_state: jnp.array):
        return jnp.linalg.norm(state[0:2] - other_state[0:2]) 