from stoix.custom_envs.cartpole_hyperplane import CartPole as CartPoleSafe
from stoix.custom_envs.cartpole_perf import CartPole as CartPolePerf

registered_envs = ["CartPoleHyperplane", "CartPolePerf"]
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
    if env_id == "CartPolePerf":
        env = CartPolePerf(**env_kwargs)
    return env, env.default_params