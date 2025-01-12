"""JAX compatible version of CartPole-v1 OpenAI gym environment."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 30.0
    tau: float = 0.02
    theta_threshold_radians: float = 24 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500  # v0 had only 200 steps!
    reward_with_bonus = True

@struct.dataclass
class EvalEnvParams(environment.EnvParams):
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 30.0
    tau: float = 0.02
    theta_threshold_radians: float = 24 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500  # v0 had only 200 steps!
    reward_with_bonus = False

class CartPole(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of CartPole-v1 OpenAI gym environment.


    Source: github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self, eval = True,mistake_trajectories=None ):
        super().__init__()
        self.obs_shape = (4,)
        self.mistake_trajectories = mistake_trajectories

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()
    @property
    def eval_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EvalEnvParams()
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        a_h = action[0]/jnp.abs(action[0])
        b_h = action[1]

        action_proposal = jax.random.uniform(key, (1), minval=-1.0, maxval=1.0)
        action_proposal_freedom = jax.random.uniform(key, (100,1), minval=-1.0, maxval=1.0)
        freedom_factor = 1-(jnp.sum(jnp.dot(a_h, action_proposal_freedom) < b_h) /100)

        def proj_fn(inp):
            a,a_h,b_h = inp
            numerator = (jnp.dot(a_h, a) - b_h)
            denominator = jnp.sum(jnp.power(a_h,2)) # ||a_h||_2^2
            projection = a - (numerator / denominator) * a_h
            return projection

        def identity_fn(inp):
            a,a_h,b_h = inp
            return a

        action = jax.lax.cond(jnp.squeeze(jnp.dot(a_h, action_proposal) < b_h), proj_fn, identity_fn, operand=(action_proposal,a_h,b_h))
        force = params.force_mag * jax.numpy.squeeze(action)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Only default Euler integration option available here!
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal*101
        
        if params.reward_with_bonus:
            reward += 0.5 * freedom_factor

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        if self.mistake_trajectories == None:
            init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
            state = EnvState(
                x=init_state[0],
                x_dot=init_state[1],
                theta=init_state[2],
                theta_dot=init_state[3],
                time=0,
            )
        else:
            num_trajectories = self.mistake_trajectories.shape[0]
            sampled_idx = jax.random.randint(key, shape=(), minval=0, maxval=num_trajectories)
            sampled_trajectory = self.mistake_trajectories[sampled_idx]
            state = EnvState(
                x=sampled_trajectory[0],
                x_dot=sampled_trajectory[1],
                theta=sampled_trajectory[2],
                theta_dot=sampled_trajectory[3],
                time=0,
            )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPoleHyperplane"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        high_action = jnp.array(1)
        return spaces.Box(-high_action, high_action, (2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )