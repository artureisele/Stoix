import wandb
import imageio
import pickle
import jax
from brax.envs.half_cheetah import Halfcheetah
def uploadVideos():
    wandb.init(project="cleanRL", name = "Videos Halfcheetah")
    #renderer = Renderer()
    with open("/home/artur/Schreibtisch/Stoix/experienced_trajectories.pkl", "rb") as file:
        loaded_list1 = pickle.load(file)
    with open("/home/artur/Schreibtisch/Stoix/experienced_actions_perf.pkl", "rb") as file:
        loaded_list2 = pickle.load(file)
    with open("/home/artur/Schreibtisch/Stoix/experienced_actions_safe.pkl", "rb") as file:
        loaded_list3 = pickle.load(file)
    env = Halfcheetah()
    for t in range(len(loaded_list1)):
        brax_state = loaded_list1[t]
        brax_state_trajectory = [jax.tree_util.tree_map(lambda x : x[i], brax_state.pipeline_state) for i in range(500)]
        frames=env.render(brax_state_trajectory)
        video_file = "trajectory.mp4"

        print(f"Video saved to {video_file}")
        imageio.mimwrite(video_file, frames, fps=30)
        wandb.log({"simulation_video": wandb.Video(video_file, format="mp4")})

if __name__ == "__main__":
    uploadVideos()