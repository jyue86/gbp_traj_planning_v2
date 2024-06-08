import jax
import jax.numpy as jnp

class FactorGraph:
    def __init__(self, start_states: jnp.array, end_pos: jnp.array) -> None:
        self.factor_to_var_msgs = None
        self.var_to_factor_msgs = None
        self.pose_likelihoods = None
        self.dynamic_likelihoods = None

        # init variable to factor messages first
    
    def run_gbp(current_states: jnp.array) -> None: 
        """
        1. Create new factor graph using current_states (and end_pos for pose factors)
        2. variable to factor messages set to uninformative
        """
        pass

    def _update_factor_to_var_messages():
        pass

    def _update_var_to_factor_messages():
        pass

    def update_factor_likelihoods():
        pass

    def _update_marginal_beliefs():
        pass