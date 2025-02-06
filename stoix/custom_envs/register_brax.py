from stoix.custom_envs.halfcheetah_hyperplane import HalfcheetahHyperplaneWrapper as HalfcheetahSafeWrapper
from stoix.custom_envs.halfcheetah_performance import HalfcheetahPerfWrapper as HalfcheetahPerfWrapper
from brax.envs.wrappers import training
from stoix.wrappers.brax import BraxJumanjiWrapper
from brax.envs import half_cheetah
registered_envs = ["HalfcheetahHyperplane", "HalfcheetahPerformance"]
def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's infamous env.make(env_name).


    Args:
      env_id: A string identifier for the environment.
      **env_kwargs: Keyword arguments to pass to the environment.


    Returns:
      A tuple of the environment and the default parameters.
    """
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered Custom environments.")

    # 1. Classic OpenAI Control Tasks
    if env_id == "HalfcheetahHyperplane":
        env = half_cheetah.Halfcheetah()
        env = training.EpisodeWrapper(env, 500, 1)
        #Modified BraxJumanjiWrapper
        env = HalfcheetahSafeWrapper(env, **env_kwargs)
        #env = BraxJumanjiWrapper(env)
    if env_id == "HalfcheetahPerformance":
        env = half_cheetah.Halfcheetah()
        env = training.EpisodeWrapper(env, 500, 1)
        env = HalfcheetahPerfWrapper(env, **env_kwargs)
        ##env = BraxJumanjiWrapper(env)
    return env