import jax
import mujoco
from mujoco import mjx
from wheelbot_test_mjx import WheelbotEnv
import jax.numpy as jnp
import imageio
a = WheelbotEnv()


mjx_model, mjx_data = a.init_mjx()
new_data = mjx_data
trajectory = [mjx_data]
for i in range(100):
    new_data, observation, info = a.step(mjx_model, new_data, jnp.array([0.1,0.1]))
    trajectory.append(new_data)
    print(observation)
    print(info)
frames = a.render(a.model, trajectory)
video_file = "jax.mp4"
print(f"Video saved to {video_file}")
imageio.mimwrite(video_file, frames, fps=1)


"""
mjx_model, mjx_data = a.init_mjx()

broadcast = lambda x: jnp.broadcast_to(x, (3, *(x.shape)))
mjx_model_batch = jax.tree_util.tree_map(broadcast, mjx_model)
mjx_data_batch = jax.tree_util.tree_map(broadcast, mjx_data)
action_batch = jnp.ones((3,2))*0.1

def step(mjx_data_batch, _):
    mjx_data_batch, obs_batch, info = jax.vmap(a.step, in_axes=(0,0,0))(mjx_model_batch, mjx_data_batch, action_batch)
    return mjx_data_batch, obs_batch
def rollout(mjx_data_batch):
    mjx_data_batch, obs_batch_traj = jax.lax.scan(
        step, mjx_data_batch, None, 200
    )
    return obs_batch_traj
result = rollout(mjx_data_batch)
print(result)
"""
#print(frames)