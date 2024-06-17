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
    
class InterRobotFactor:
    def __init__(
        self,
        state: jnp.ndarray,
        critical_distance: float,
        t: jnp.ndarray, #ndarray just to hold time,
        dims: jnp.ndarray,
    ) -> None:
        self._crit_distance = critical_distance
        self._z_precision = 15 # 100 

        self._dist = self._calc_dist(state[0:4], state[4:]) 
        # jax.debug.print("dist: {}", self._dist)
        # jax.debug.breakpoint()
        dx, dy = (state[0] - state[4])/self._dist, (state[1] - state[5])/self._dist
        self._J = jnp.array([[-dx/self._crit_distance, -dy/self._crit_distance, 0, 0,
                              dx/self._crit_distance, dy/self._crit_distance, 0, 0]])

        self._state = state
        # self._state_precision = self._z_precision * jnp.eye(1) * (self._crit_distance ** 2)
        self._state_precision = (t * INTER_ROBOT_NOISE) ** -2 * jnp.eye(1)
        self._energy_precision = (t * INTER_ROBOT_NOISE) ** -2 * jnp.eye(4)
        self._dims = dims

        self._gap_multiplier = 1e10 # 1e3
    
    def calculate_likelihood(self) -> Gaussian:
        return Gaussian(
            self._calc_info(self._state, self._state_precision),
            self._calc_precision(self._state, self._state_precision),
            self._dims
        )
 
    def _calc_dist(self, state: jnp.array, other_state: jnp.array):
        return jnp.linalg.norm((state[0:2] - other_state[0:2]) + 1e-6)
    
    def _calc_info(self, state: jnp.ndarray, state_precision: jnp.ndarray) -> jnp.ndarray:
        def safe_fn():
            return jnp.zeros(8) # state_precision @ state[jnp.newaxis,:]).squeeze()
        def unsafe_fn():
            return self._gap_multiplier * (self._J.T @ state_precision @ (self._J @ state[:,jnp.newaxis] - self._calc_measurement(state))).squeeze()
        info = jax.lax.select(self._dist >= self._crit_distance, safe_fn(), unsafe_fn())
        # jax.debug.print("Info: {}", info)
        # jax.debug.breakpoint()
        return info
        
    def _calc_precision(self, state: jnp.ndarray, state_precision: jnp.ndarray) -> jnp.ndarray:
        unsafe_precision = self._J.T @ state_precision @ self._J 
        # unsafe_precision = unsafe_precision.at[2:4,2:4].set(jnp.eye(2)).at[6:,6:].set(jnp.eye(2))

        # Update A
        # unsafe_precision = unsafe_precision.at[:2,:2].set(unsafe_precision[0, 0])

        # # Update B
        # unsafe_precision = unsafe_precision.at[4:6,4:6].set(unsafe_precision[4,4])

        # # Update C
        # unsafe_precision = unsafe_precision.at[0:2, 4:6].set(unsafe_precision[0, 4])

        # # Update D
        # unsafe_precision = unsafe_precision.at[4:6, 0:2].set(unsafe_precision[4, 0])

        # diag_values = jnp.diag(unsafe_precision)
        # unsafe_precision = jnp.fill_diagonal(unsafe_precision, jnp.where(diag_values == 0, 1.0, diag_values), inplace=False)

        # eye to enter the cycle
        # zeros to go back to "the working branch"
        # precision = jax.lax.select(self._dist >= self._crit_distance, jnp.zeros_like(unsafe_precision), unsafe_precision)
        # jax.debug.print("precision: {}", precision)
        # jax.debug.print("inv precision: {}", jnp.linalg.inv(precision))
        # jax.debug.breakpoint()

        return unsafe_precision # precision

    def _calc_measurement(self, state: jnp.ndarray):
        current_state = state[0:4]
        other_state = state[4:]
        dist = self._calc_dist(current_state, other_state) + 1e-6
        measurement = jax.lax.select(
            dist < self._crit_distance, 1.0 - dist / self._crit_distance, 0.
        )
        return measurement

class ObstacleFactor:
    def __init__(
        self, state: jnp.ndarray, closest_obstacle: jnp.ndarray, crit_distance: float, agent_radius: float, dims: jnp.ndarray
    ) -> None:
        self._state = state
        self._closest_obstacle = closest_obstacle
        self._crit_distance = crit_distance
        self._agent_radius = agent_radius
        self._state_precision = OBSTACLE_NOISE ** (-2) * jnp.eye(1)
        self._energy_precision = (OBSTACLE_NOISE) ** -2 * jnp.eye(4)
        self._dims = dims
        self._gap_multiplier = 1

        dist, dx, dy = self._calc_dist(state, closest_obstacle) 
        def safe_fn():
            return jnp.zeros((1,4))
        def unsafe_fn():
            return jnp.array([[-dx/crit_distance, -dy/crit_distance, 0, 0]])
        self._J = jax.lax.select(dist >= self._crit_distance, safe_fn(), unsafe_fn())
    
    def calculate_likelihood(self) -> Gaussian:
        return Gaussian(
            self._calc_info(self._state, self._state_precision),
            self._calc_precision(self._state_precision),
            self._dims
        )

    def _calc_info(self, state: jnp.ndarray, state_precision: jnp.ndarray):
        info = self._gap_multiplier * (self._J.T @ state_precision @ (self._J @ state[:,jnp.newaxis] - self._calc_measurement(state))).squeeze()
        return info

    def _calc_precision(self, state_precision: jnp.ndarray):
        precision = self._J.T @ state_precision @ self._J
        return precision

    def _calc_measurement(self, state):
        dist = self._calc_dist(state, self._closest_obstacle)[0]
        return jax.lax.select(dist < self._agent_radius, 1 - dist / self._agent_radius, 0.)

    def _calc_dist(self, state, other_state):
        dist = jnp.linalg.norm(state[0:2] - other_state[0:2]) - self._agent_radius
        return dist, state[0] - other_state[0], state[1] - other_state[1]