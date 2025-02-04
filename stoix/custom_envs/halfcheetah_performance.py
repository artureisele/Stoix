from typing import Any, Dict, Optional, Tuple

import chex
import jax.numpy as jnp
from brax import base
from brax.envs.base import Wrapper as BraxWrapper
from flax import struct
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.types import StepType, TimeStep, restart
import jax
from stoix.base_types import Observation


@struct.dataclass
class BraxState(base.Base):
    pipeline_state: Optional[base.State]
    obs: chex.Array
    reward: chex.Numeric
    done: chex.Numeric
    key: chex.PRNGKey
    step_count: chex.Array
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class HalfcheetahPerfWrapper(BraxWrapper):
    def __init__(
        self,
        env: Environment,
        safety_filter_function = None,
        safety_filter_params = None,
        safe_filter_q = None,
        safe_filter_q_params = None
    ):
        """Initialises a Brax wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self._env = env
        self._action_dim = self.action_spec().shape[0]
        self.safety_filter_function = safety_filter_function
        self.safety_filter_params = safety_filter_params
        self.safe_filter_q = safe_filter_q
        self.safe_filter_q_params = safe_filter_q_params

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:

        state = self._env.reset(key)

        new_state = BraxState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            key=key,
            metrics=state.metrics,
            info=state.info,
            step_count=jnp.array(0, dtype=int),
        )
        agent_view = new_state.obs.astype(float)
        legal_action_mask = jnp.ones((self._action_dim,), dtype=float)

        timestep = restart(
            observation=Observation(
                agent_view,
                legal_action_mask,
                new_state.step_count,
            ),
            extras={"q_safe_value":-150, "safe_action":jnp.zeros((7), ),"filter_factor":-1.0},
        )

        return new_state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        action_proposal = action
        if self.safety_filter_function!=None and self.safety_filter_params!=None:
            observation = state.obs
            q_value_safety = self.safe_filter_q(self.safe_filter_q_params, observation)
            action_safety_filter = self.safety_filter_function(
                self.safety_filter_params,
                observation,
                key,
            )
            a_h = action_safety_filter[:-1]/jnp.linalg.norm(action_safety_filter[:-1])
            b_h = action_safety_filter[-1]*jnp.pow(6,0.5)
            #jax.debug.print("Safe a_h :{}",a_h)
            #jax.debug.print("Safe b_h :{}",b_h)
            def proj_fn(inp):
                a,a_h,b_h = inp
                numerator = (jnp.dot(a_h, a) - b_h)
                denominator = jnp.power(jnp.linalg.norm(action[:-1]),2) # ||a_h||_2^2
                projection = a - (numerator / denominator) * a_h
                return projection

            def identity_fn(inp):
                a,a_h,b_h = inp
                return a
            action_proposal_freedom = jax.random.uniform(key, (100,6), minval=-1.0, maxval=1.0)
            #filter_factor = (jnp.max(jnp.array([jnp.sum(jnp.dot(a_h, action_proposal_freedom) < b_h) /100,0.4]))-0.4)/2
            action = jax.lax.cond(jnp.squeeze(jnp.dot(a_h, action_proposal) < b_h), proj_fn, identity_fn, operand=(action_proposal,a_h,b_h)) 

        # If the previous step was truncated
        prev_truncated = state.info["truncation"].astype(jnp.bool_)
        # If the previous step was done
        prev_terminated = state.done.astype(jnp.bool_)
        key, key2 = jax.random.split(state.key, 2)
        state = self._env.step(state, action, key2)


        # This is true only if truncated
        truncated = state.info["truncation"].astype(jnp.bool_)
        # This is true if truncated or done
        terminated = jax.lax.cond(jnp.logical_or(state.obs[0]< -0.3,truncated), lambda: True, lambda: False)
        state = BraxState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=terminated.astype(float),
            key=state.key,
            metrics=state.metrics,
            info=state.info,
            step_count=state.step_count + 1,
        )
        
        # If terminated make the discount zero, otherwise one
        discount = jnp.where(terminated, 0.0, 1.0)
        # However, if truncated, make the discount one
        discount = jnp.where(truncated, 1.0, discount)
        # Lastly, if the previous step was truncated or terminated, make the discount zero
        # This is to ensure that the discount is zero for the last step of the episode
        # and that stepping past the last step of the episode does not affect the discount
        discount = jnp.where(prev_truncated | prev_terminated, 0.0, discount)

        # If terminated or truncated step type is last, otherwise mid
        step_type = jnp.where(terminated | truncated, StepType.LAST, StepType.MID)

        agent_view = state.obs.astype(float)
        legal_action_mask = jnp.ones((self._action_dim,), dtype=float)
        obs = Observation(
            agent_view,
            legal_action_mask,
            state.step_count,
        )

        next_timestep = TimeStep(
            step_type=step_type,
            reward=state.reward.astype(float),
            discount=discount.astype(float),
            observation=obs,
            extras={"q_safe_value": q_value_safety,
             "safe_action": action_safety_filter,
             "filter_factor": jnp.sum(jnp.abs(action-action_proposal)),}, #filter_factor
        )

        return state, next_timestep

    def action_spec(self) -> specs.Spec:
        action_space = specs.BoundedArray(
            shape=(self.action_size,),
            dtype=float,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )
        return action_space

    def observation_spec(self) -> specs.Spec:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=specs.Array(shape=(self.observation_size,), dtype=float),
            action_mask=specs.Array(shape=(self.action_size,), dtype=float),
            step_count=specs.Array(shape=(), dtype=int),
        )

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount")
