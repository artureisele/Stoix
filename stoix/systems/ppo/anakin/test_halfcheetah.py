from stoix.custom_envs.halfcheetah_hyperplane import HalfcheetahHyperplaneWrapper as HalfcheetahSafeWrapper
from stoix.custom_envs.halfcheetah_performance import HalfcheetahPerfWrapper as HalfcheetahPerfWrapper
from brax.envs.wrappers import training
from stoix.wrappers.brax import BraxJumanjiWrapper
from brax.envs import half_cheetah
from jumanji.wrappers import AutoResetWrapper
from stoix.wrappers import GymnaxWrapper, JumanjiWrapper, RecordEpisodeMetrics
import jax.random as random
env = half_cheetah.Halfcheetah()
env = training.EpisodeWrapper(env, 500, 1)
#Modified BraxJumanjiWrapper
env = HalfcheetahSafeWrapper(env)
env = AutoResetWrapper(env, next_obs_in_extras=True)
env = RecordEpisodeMetrics(env)

key = random.PRNGKey(42)
key, reset_key,action_key = random.split(key, 3)
env_state, timestep = env.reset(reset_key)
action = random.uniform(action_key, shape=(env.action_size,), minval=-1.0, maxval=1.0)
def step_fn(state, _):