from stoix.custom_envs.cartpole_hyperplane import CartPole as CartPoleSafe
from stoix.custom_envs.cartpole_performance import CartPole as CartPolePerf
from stoix.custom_envs.wheelbot_test_mjx import WheelbotEnv as Wheelbot

registered_envs = ["CartPoleHyperplane", "CartPolePerformance", "Wheelbot-v0"]
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
    if env_id == "CartPoleHyperplane":
        env = CartPoleSafe(**env_kwargs)
    if env_id == "CartPolePerformance":
        env = CartPolePerf(**env_kwargs)
    if env_id == "Wheelbot-v0":
        env = Wheelbot(**env_kwargs)
    return env, env.default_params