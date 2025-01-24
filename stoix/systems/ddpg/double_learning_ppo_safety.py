import copy
import time
import os
import sys
from typing import Any, Callable, Dict, Tuple
new_project_path = os.path.dirname(os.path.abspath(__file__))
if new_project_path not in sys.path:
    sys.path.insert(0, new_project_path)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import chex
import flashbax as fbx
import flax
import hydra
import jax
import math
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
import pickle
from stoix.systems.ddpg.upload_videos_cartpole_to_wandb import uploadVideos

from stoix.base_types import (
    ActorApply,
    AnakinExperimentOutput,
    ContinuousQApply,
    LearnerFn,
    LogEnvState,
    Observation,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import CompositeNetwork
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import MultiNetwork
from stoix.networks.postprocessors import tanh_to_spec
from stoix.systems.ddpg.ddpg_types import DDPGOptStates, DDPGParams
from stoix.systems.q_learning.dqn_types import Transition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics

from stoix.systems.ppo.anakin.ff_ppo_continuous_kl_early_stopping import learner_setup as learner_setup_s
from stoix.systems.ppo.anakin.ff_ppo_continuous_kl_early_stopping import get_learner_fn as get_learner_fn_s
#from stoix.systems.ppo.anakin.ff_ppo_continuous_kl_early_stopping import get_warmup_fn as get_warmup_fn_s
#from stoix.systems.ppo.anakin.ff_ppo_continuous_kl_early_stopping import get_default_behavior_policy as get_default_behavior_policy_s

from stoix.systems.sac.ff_sac_delayed import learner_setup as learner_setup_p
from stoix.systems.sac.ff_sac_delayed import get_learner_fn as get_learner_fn_p
from stoix.systems.sac.ff_sac_delayed import get_warmup_fn as get_warmup_fn_p



def initialize_config_dicts(_config_s, _config_p):
    # Calculate total timesteps.
    n_devices = len(jax.devices())

    config_s = copy.deepcopy(_config_s)
    config_s.n_devices = n_devices

    assert config_s.arch.total_num_envs % (config_s.n_devices * config_s.arch.update_batch_size) == 0, (
    f"{Fore.RED}{Style.BRIGHT}The total number of environments "
    + f"should be divisible by the n_devices*update_batch_size!{Style.RESET_ALL}"
    )
    # Number of environments per device
    config_s.arch.num_envs = int(
        config_s.arch.total_num_envs // (config_s.n_devices * config_s.arch.update_batch_size)
    )
    # Set number of updates per evaluation and logging
    config_s.arch.num_updates_per_eval = config_s.system.safe_num_updates_per_eval 

    config_s.system.steps_per_rollout = (
        config_s.n_devices
        * config_s.arch.num_updates_per_eval
        * config_s.system.rollout_length
        * config_s.arch.update_batch_size
        * config_s.arch.num_envs
    )

    config_p = copy.deepcopy(_config_p)
    config_p.n_devices = n_devices
    # Number of environments per device
    config_p.arch.num_envs = int(
        config_p.arch.total_num_envs // (config_p.n_devices * config_p.arch.update_batch_size)
    ) 
    config_p.arch.num_updates_per_eval=1

    config_p.system.steps_per_rollout=(
        config_p.n_devices
        * config_p.arch.num_updates_per_eval
        * config_p.system.rollout_length
        * config_p.arch.update_batch_size
        * config_p.arch.num_envs
    )
    #Initial Learning shall get transition only from safe uniform action sampling->take rollout_length to 0 for one time
    config_p.system.rollout_length = 0
    config_p.system.epochs = config_p.system.initial_learning_epochs
    # PRNG keys, seed are the same for both configs, choose config_s as default config
    key = jax.random.PRNGKey(config_s.arch.seed)
    config_p.arch.evaluation_greedy = False #Important, that when we check if the filtered performance policy is safe, we evaluate with samples like in the "real" env SAC agent
    return config_s, config_p, key

def generate_safety_env_and_learn(config_s, key, custom_extras):
    key,safe_actor_net_key,safe_q_net_key, safe_key_e=jax.random.split(key, num=4)
    # Create the environments for train and eval for Safety Agent.
    safe_env, safe_eval_env = environments.make(config=config_s, custom_extras=custom_extras)
    
    # Setup safe learner
    safe_learn, safe_actor_network,safe_q_network, safe_learner_state = learner_setup_s(
        safe_env, (key, safe_actor_net_key, safe_q_net_key), config_s
    )

    # Setup safety evaluator.
    safe_evaluator, _, (safe_trained_params, eval_keys) = evaluator_setup(
        eval_env=safe_eval_env,
        key_e=safe_key_e,
        eval_act_fn=get_distribution_act_fn(config_s, safe_actor_network.apply),
        params=safe_learner_state.params.actor_params,
        config=config_s,
    )
    return key, safe_learn, safe_actor_network, safe_q_network, safe_learner_state, safe_evaluator

def log_training_metrics_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, safe_learner_output, logger):
    # Log the results of the training.
    t = int(config_p.system.steps_per_rollout * (eval_step_perf + 1)) + int(config_s.system.steps_per_rollout * (eval_step_safety + 1))
    eval_step = eval_step_perf+eval_step_safety
    episode_metrics, ep_completed = get_final_step_metrics(safe_learner_output.episode_metrics)
    episode_metrics["steps_per_second"] = config_s.system.steps_per_rollout / elapsed_time
    # Separately log timesteps, actoring metrics and training metrics.
    logger.log({"timestep": t}, t, eval_step, LogEvent.MISC_SAFE)
    if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
        logger.log(episode_metrics, t, eval_step, LogEvent.ACT_SAFE)

    train_metrics = safe_learner_output.train_metrics
    # Calculate the number of optimiser steps per second. Since gradients are aggregated
    # across the device and batch axis, we don't consider updates per device/batch as part of
    # the SPS for the learner.
    opt_steps_per_eval = config_s.arch.num_updates_per_eval * (config_s.system.epochs)
    train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
    logger.log(train_metrics, t, eval_step, LogEvent.TRAIN_SAFE)

def prepare_safe_evaluation(config_s, key, safe_learner_output):
    safe_trained_params = unreplicate_batch_dim(
        safe_learner_output.learner_state.params.actor_params
    )  # Select only actor params
    key, *eval_keys = jax.random.split(key, config_s.n_devices + 1)
    eval_keys = jnp.stack(eval_keys).reshape(config_s.n_devices, -1)

    return key, safe_trained_params, eval_keys

def log_evaluation_metrics_safety_training(config_s,config_p, elapsed_time, eval_step_safety, eval_step_perf, safe_evaluator_output,logger):
    episode_return = jnp.mean(safe_evaluator_output.episode_metrics["episode_return"])
    t = int(config_p.system.steps_per_rollout * (eval_step_perf + 1)) + int(config_s.system.steps_per_rollout * (eval_step_safety + 1))
    eval_step = eval_step_perf+eval_step_safety
    steps_per_eval = int(jnp.sum(safe_evaluator_output.episode_metrics["episode_length"]))
    safe_evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    logger.log(safe_evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL_SAFE)
    return episode_return

def generate_performance_learner_and_evaluator(config_s, key, config_p, safe_actor_network, safe_q_network, safe_learner_state):
    custom_extras = {
        "safety_filter_function" : get_distribution_act_fn(config_s, safe_actor_network.apply),
        "safety_filter_params": unreplicate_n_dims(safe_learner_state.params.actor_params),
        "safe_filter_q": safe_q_network.apply,
        "safe_filter_q_params":unreplicate_n_dims(safe_learner_state.params.critic_params),
    }
    perf_env, perf_eval_env = environments.make(config=config_p, custom_extras = custom_extras)

    key, perf_actor_net_key, perf_q_net_key, perf_key_e = jax.random.split(key, 4)
    #Here the replay buffer is getting filled with filtered initial random SAC Policy
    perf_learn, perf_actor_network, perf_learner_state = learner_setup_p(
        perf_env, (key, perf_actor_net_key, perf_q_net_key), config_p
    )

    # Setup performance evaluator.
    perf_evaluator, perf_absolute_metric_evaluator, (perf_trained_params, eval_keys) = evaluator_setup(
        eval_env=perf_eval_env,
        key_e=perf_key_e,
        eval_act_fn=get_distribution_act_fn(config_p, perf_actor_network.apply),
        params=perf_learner_state.params.actor_params,
        track_failed_trajectories = True,
        config=config_p,
    )
    return key, perf_learn, perf_actor_network, perf_learner_state, perf_evaluator

def log_training_metrics_performance_training(config_p,config_s, elapsed_time, eval_step_perf,eval_step_safety, perf_learner_output, logger):
    # Log the results of the training.
    t = int(config_p.system.steps_per_rollout * (eval_step_perf + 1)) + int(config_s.system.steps_per_rollout * (eval_step_safety + 1))
    eval_step = eval_step_perf+eval_step_safety
    episode_metrics, ep_completed = get_final_step_metrics(perf_learner_output.episode_metrics)
    episode_metrics["steps_per_second"] = config_p.system.steps_per_rollout / elapsed_time

    # Separately log timesteps, actoring metrics and training metrics.
    logger.log({"timestep": t}, t, eval_step, LogEvent.MISC_Perf)
    if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
        logger.log(episode_metrics, t, eval_step, LogEvent.ACT_Perf)
    train_metrics = perf_learner_output.train_metrics
    # Calculate the number of optimiser steps per second. Since gradients are aggregated
    # across the device and batch axis, we don't consider updates per device/batch as part of
    # the SPS for the learner.
    opt_steps_per_eval = 1 * (config_p.system.epochs)
    train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
    logger.log(train_metrics, t, eval_step, LogEvent.TRAIN_Perf)

def prepare_performance_evaluation(config_p, key, perf_learner_output):
    perf_trained_params = unreplicate_batch_dim(
        perf_learner_output.learner_state.params.actor_params
    )  # Select only actor params
    key, *perf_eval_keys = jax.random.split(key, config_p.n_devices + 1)
    perf_eval_keys = jnp.stack(perf_eval_keys)
    perf_eval_keys = perf_eval_keys.reshape(config_p.n_devices, -1)
    return key, perf_trained_params, perf_eval_keys

def log_evaluation_metrics_performance_agent_for_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, perf_evaluator_output,logger):
    t = int(config_p.system.steps_per_rollout * (eval_step_perf + 1)) + int(config_s.system.steps_per_rollout * (eval_step_safety + 1))
    eval_step = eval_step_perf+eval_step_safety
    steps_per_eval = int(jnp.sum(perf_evaluator_output.episode_metrics["episode_length"]))
    perf_evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    logger.log(perf_evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL_Perf)

    perf_eval_length = jnp.average(perf_evaluator_output.episode_metrics["episode_length"])
    return perf_eval_length

def log_evaluation_metrics_performance_agent_greedy(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, perf_evaluator_output,logger):
    t = int(config_p.system.steps_per_rollout * (eval_step_perf + 1)) + int(config_s.system.steps_per_rollout * (eval_step_safety + 1))
    eval_step = eval_step_perf+eval_step_safety
    steps_per_eval = int(jnp.sum(perf_evaluator_output.episode_metrics["episode_length"]))
    perf_evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    logger.log(perf_evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL_Perf_Greed)

    perf_eval_length = jnp.average(perf_evaluator_output.episode_metrics["episode_length"])
    return perf_eval_length

def plot_border_decisions(ev):
    fig = plt.figure(figsize=(8, 8), clear=True, num=0)
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((-2.4, -24 * 2 * math.pi / 360), 2 * 2.4, 2 * 24 * 2 * math.pi / 360,
                        linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    quiver_plot = ax.quiver(ev["_x_grid"], ev["_z_grid"], ev["_arrowDirX"], ev["_arrowDirY"], ev["_threshold"], cmap="viridis",
                            angles="xy", scale_units="xy", scale=25)
    # Set the plot boundaries
    plt.axis([-2 * 2.4, 2 * 2.4, -2 * 24 * 2 * math.pi / 360, 2 * 24 * 2 * math.pi / 360])
    plt.colorbar(quiver_plot, label="Safe actions from value in direction of arrow")
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Cartpole Border Decisions")
    # Show the plot
    img = wandb.Image(plt)
    #wandb.log({"test":img})
    ev["Cartpole_Border_Decisions"] = img

def plot_eval_trajectories(perf_evaluator_output):
    fig = plt.figure(figsize=(8, 8), clear=True, num=0)
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((-2.4, -24 * 2 * math.pi / 360), 2 * 2.4, 2 * 24 * 2 * math.pi / 360,
                        linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    all_action_taken_performance = perf_evaluator_output.action_taken_performance[0]
    all_action_taken_safety= perf_evaluator_output.action_taken_safety[0]
    all_trajectories = perf_evaluator_output.trajectories[0]
    safe_v_values = perf_evaluator_output.safe_q_values[0]
    all_trajectories_reshaped = all_trajectories.reshape(-1,4)
    safe_v_values_reshaped = safe_v_values.reshape(-1,1)
    indices = jnp.any(all_trajectories_reshaped!=0,axis=1)
    all_trajectories_x = all_trajectories_reshaped[indices][:,0].tolist()
    all_trajectories_theta = all_trajectories_reshaped[indices][:,2].tolist()
    safe_v_values_without_zeros = safe_v_values_reshaped[indices]
    # Set the plot boundaries
    eval_traj_plot = plt.scatter(all_trajectories_x, all_trajectories_theta, c=safe_v_values_without_zeros, cmap="viridis", s=5)
    plt.axis([-2 * 2.4, 2 * 2.4, -2 * 24 * 2 * math.pi / 360, 2 * 24 * 2 * math.pi / 360])
    plt.colorbar(eval_traj_plot, label="V Value")
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Eval Trajectories all")
    # Show the plot
    img = wandb.Image(plt)
    #wandb.log({"test":img})
    perf_evaluator_output.episode_metrics["Eval_Trajectories"] = img

def plot_eval_trajectories_candidates_mistake_trajectories(perf_evaluator_output):
    fig = plt.figure(figsize=(8, 8), clear=True, num=0)
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((-2.4, -24 * 2 * math.pi / 360), 2 * 2.4, 2 * 24 * 2 * math.pi / 360,
                        linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)

    all_action_taken_performance = perf_evaluator_output.action_taken_performance[0]
    all_action_taken_safety= perf_evaluator_output.action_taken_safety[0]
    all_trajectories = perf_evaluator_output.trajectories[0]
    safe_v_values = perf_evaluator_output.safe_q_values[0]

    relevant_ep_lengths = perf_evaluator_output.episode_metrics["episode_length"][0,:,0][ perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
    safe_v_values= safe_v_values[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
    all_trajectories = all_trajectories[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
    all_trajectories = all_trajectories[jnp.arange(all_trajectories.shape[0]),jnp.clip((relevant_ep_lengths-25),a_min=0).astype(int)]
    safe_v_values = safe_v_values[jnp.arange(safe_v_values.shape[0]),jnp.clip((relevant_ep_lengths-25),a_min=0).astype(int)]
    #jax.debug.breakpoint()


    all_trajectories_reshaped = all_trajectories.reshape(-1,4)
    safe_v_values_reshaped = safe_v_values.reshape(-1,1)
    all_trajectories_x = all_trajectories_reshaped[:,0].tolist()
    all_trajectories_theta = all_trajectories_reshaped[:,2].tolist()
    safe_v_values_without_zeros = safe_v_values_reshaped.tolist()
    # Set the plot boundaries
    eval_traj_plot = plt.scatter(all_trajectories_x, all_trajectories_theta, c=safe_v_values_without_zeros, cmap="viridis", s=10)
    plt.axis([-2 * 2.4, 2 * 2.4, -2 * 24 * 2 * math.pi / 360, 2 * 24 * 2 * math.pi / 360])
    plt.colorbar(eval_traj_plot, label="V Value")
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Eval Trajectories Candidates End 25")
    # Show the plot
    img = wandb.Image(plt)
    #wandb.log({"test":img})
    perf_evaluator_output.episode_metrics["Eval_Trajectories_End_25"] = img

def plot_eval_trajectories_candidates_mistake_trajectories2(perf_evaluator_output):
    fig = plt.figure(figsize=(8, 8), clear=True, num=0)
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((-2.4, -24 * 2 * math.pi / 360), 2 * 2.4, 2 * 24 * 2 * math.pi / 360,
                        linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)

    all_action_taken_performance = perf_evaluator_output.action_taken_performance[0]
    all_action_taken_safety= perf_evaluator_output.action_taken_safety[0]
    all_trajectories = perf_evaluator_output.trajectories[0]
    safe_v_values = perf_evaluator_output.safe_q_values[0]

    relevant_ep_lengths = perf_evaluator_output.episode_metrics["episode_length"][0,:,0][ perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
    safe_v_values= safe_v_values[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
    all_trajectories = all_trajectories[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
    all_trajectories = all_trajectories[jnp.arange(all_trajectories.shape[0]),jnp.clip((relevant_ep_lengths-50),a_min=0).astype(int)]
    safe_v_values = safe_v_values[jnp.arange(safe_v_values.shape[0]),jnp.clip((relevant_ep_lengths-50),a_min=0).astype(int)]


    all_trajectories_reshaped = all_trajectories.reshape(-1,4)
    safe_v_values_reshaped = safe_v_values.reshape(-1,1)
    all_trajectories_x = all_trajectories_reshaped[:,0].tolist()
    all_trajectories_theta = all_trajectories_reshaped[:,2].tolist()
    safe_v_values_without_zeros = safe_v_values_reshaped.tolist()
    # Set the plot boundaries
    eval_traj_plot = plt.scatter(all_trajectories_x, all_trajectories_theta, c=safe_v_values_without_zeros, cmap="viridis", s=10)
    plt.axis([-2 * 2.4, 2 * 2.4, -2 * 24 * 2 * math.pi / 360, 2 * 24 * 2 * math.pi / 360])
    plt.colorbar(eval_traj_plot, label="V Value")
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Eval Trajectories Candidates End 50")
    # Show the plot
    img = wandb.Image(plt)
    #wandb.log({"test":img})
    perf_evaluator_output.episode_metrics["Eval_Trajectories_End_50"] = img

def run_experiment(_config_s: DictConfig, _config_p: DictConfig) -> float:
    """Runs experiment."""
    print("Initialize the config dicts!")
    config_s,config_p, key = initialize_config_dicts(_config_s,_config_p)
    # Logger setup
    logger = StoixLogger(_config_s)

    print("Start initalizing safety training!")
    key, safe_learn, safe_actor_network, safe_q_network, safe_learner_state, safe_evaluator = generate_safety_env_and_learn(config_s, key, custom_extras={})

    print("Start first safety training for regular starting states")
    safe = False
    eval_step_safety = 0
    eval_step_perf=0
    perf_eval_best_reward = 0
    maximal_trajectories=[]
    maximal_actions_taken_performance=[]
    maximal_actions_taken_safety=[]
    while(not safe):
        start_time = time.time()
        print(f"Start safety learning for eval_step: {eval_step_safety}")
        safe_learner_output = safe_learn(safe_learner_state)
        jax.block_until_ready(safe_learner_output)
        elapsed_time = time.time() - start_time
        
        print(f"Start logging training results for eval_step: {eval_step_safety}")
        log_training_metrics_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, safe_learner_output, logger)

        start_time = time.time()
        key, safe_trained_params, eval_keys = prepare_safe_evaluation(config_s, key, safe_learner_output)
        # Evaluate. This determines if we can stop the safe training
        print(f"Start safety evaluation for eval_step: {eval_step_safety}")
        safe_evaluator_output = safe_evaluator(safe_trained_params, eval_keys)
        jax.block_until_ready(safe_evaluator_output)

        plot_border_decisions(safe_evaluator_output.episode_metrics)




        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = log_evaluation_metrics_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, safe_evaluator_output,logger)
        print(f"Episode Return of evaluation: {episode_return}")

        # Terminate if environment is safety filtered
        if episode_return>= config_s.env.solved_return_threshold:
            safe = True

        eval_step_safety +=1
        # Update runner state to continue training.
        safe_learner_state = safe_learner_output.learner_state

    print("Start Initializing the Performance Training")
    key, perf_learn, perf_actor_network, perf_learner_state, perf_evaluator=generate_performance_learner_and_evaluator(config_s,key,config_p, safe_actor_network,safe_q_network, safe_learner_state)
    
    print("Start first performance training for biasing the first learned performance polciy")
    start_time = time.time()
    perf_learner_output = perf_learn(perf_learner_state)
    jax.block_until_ready(perf_learner_output)

    #Return to original values, high rollout length and epochs with just one update step to first gather safe experience and then learn from all of it
    config_p.system.rollout_length = _config_p.system.rollout_length
    config_p.system.epochs = _config_p.system.epochs

    elapsed_time = time.time() - start_time    
    log_training_metrics_performance_training(config_p,config_s, elapsed_time, eval_step_perf,eval_step_safety, perf_learner_output, logger)
    
    eval_step_perf+=1
    # Update runner state to continue training.
    perf_learner_state = perf_learner_output.learner_state

    perf_eval_length = 0
    # Prepare for evaluation.
    for i in range(config_s.number_double_learning_iterations):
        print(f"Start {i}th round of Double Learning")
        safety_assured_counter = 0
        while True:

            key, perf_learn, _, _, perf_evaluator=generate_performance_learner_and_evaluator(config_s,key,config_p, safe_actor_network,safe_q_network, safe_learner_state)

            key, perf_trained_params, perf_eval_keys = prepare_performance_evaluation(config_p, key, perf_learner_output)

            start_time = time.time()
            # Evaluate.
            perf_evaluator_output = perf_evaluator(perf_trained_params, perf_eval_keys)
            jax.block_until_ready(perf_evaluator_output)
            plot_eval_trajectories(perf_evaluator_output)
            plot_eval_trajectories_candidates_mistake_trajectories2(perf_evaluator_output)
            plot_eval_trajectories_candidates_mistake_trajectories(perf_evaluator_output)
            # Log the results of the evaluation.
            elapsed_time = time.time() - start_time
            perf_eval_length = log_evaluation_metrics_performance_agent_for_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, perf_evaluator_output,logger)
            print(f"Performance eval length: {perf_eval_length}")

            """
            if perf_eval_length >=config_s.env.solved_return_threshold:
                safety_assured_counter +=1
                all_trajectories = perf_evaluator_output.trajectories
                argmax_eval_index = perf_evaluator_output.episode_metrics["episode_return"].argmax()
                best_trajectory = all_trajectories[0][argmax_eval_index]
                if len(best_trajectory)>40:
                    best_trajectory = best_trajectory[:(best_trajectory.shape[0]//10)*10,:]
                    best_trajectory = best_trajectory[:-30,:]
                    section_size = best_trajectory.shape[0]/10
                    starting_indices = jnp.arange(0,best_trajectory.shape[0], section_size)[:10]
                    offset = jax.random.choice(key, section_size, shape=(10,), replace=True)
                    final_mistake_trajectories = best_trajectory[(starting_indices+offset).astype(int)]
                elif len(best_trajectory)>10:
                    random_indices = jax.random.choice(key, best_trajectory.shape[0], shape=(10,), replace=False)
                    final_mistake_trajectories= best_trajectory[random_indices]
            else:#Case where we shall learn from mistakes
                #Misuse variable here of learner_state of perf_evaluator for the observation experienced in failed trajectories
                safety_assured_counter = 0
                final_mistake_trajectories = perf_evaluator_output.learner_state
                final_mistake_trajectories_flatten = final_mistake_trajectories[0]
                safe_q_values = perf_evaluator_output.safe_q_values
                safe_q_values_relevant = safe_q_values[0]
                episode_lengths_smaller500 = perf_evaluator_output.episode_metrics["episode_length"][0,:,0][perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
                final_mistake_trajectories_flatten = final_mistake_trajectories_flatten[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
                shortest_row_index = episode_lengths_smaller500.argsort()[int(episode_lengths_smaller500.shape[0]//2)]
                final_mistake_trajectories = final_mistake_trajectories_flatten[shortest_row_index]
                safe_q_values_relevant = safe_q_values_relevant[shortest_row_index]
                indices = jnp.any(final_mistake_trajectories!=0,axis=1)
                print(f"Length median chain{len(final_mistake_trajectories[indices])}")

                final_mistake_trajectories = final_mistake_trajectories[indices]
                safe_q_values_relevant = safe_q_values_relevant[indices]
                if len(final_mistake_trajectories)!=0:
                    def softmax(x):
                        exp_x = jnp.exp(x - jnp.max(x))  # Subtract max for numerical stability
                        return exp_x / exp_x.sum()
                    if len(final_mistake_trajectories)>55:
                        final_mistake_trajectories = final_mistake_trajectories[:(final_mistake_trajectories.shape[0]//5)*5,:]
                        final_mistake_trajectories = final_mistake_trajectories[:-50,:]
                        section_size = final_mistake_trajectories.shape[0]/5
                        starting_indices = jnp.arange(0,final_mistake_trajectories.shape[0], section_size)[:5]
                        offset = jax.random.choice(key, section_size, shape=(5,), replace=True)
                        final_mistake_trajectories = final_mistake_trajectories[(starting_indices+offset).astype(int)]
                    elif len(final_mistake_trajectories)>5:
                        random_indices = jax.random.choice(key, final_mistake_trajectories.shape[0], shape=(5,), replace=False)
                        final_mistake_trajectories= final_mistake_trajectories[random_indices]
            """
            final_mistake_trajectories = perf_evaluator_output.learner_state[0]
            safe_q_values = perf_evaluator_output.safe_q_values[0]

            #safe_q_values_relevant = safe_q_values[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
            relevant_ep_lengths = perf_evaluator_output.episode_metrics["episode_length"][0,:,0][ perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
            final_mistake_trajectories = final_mistake_trajectories[perf_evaluator_output.episode_metrics["episode_length"][0,:,0]<500]
            final_mistake_trajectories_ends = final_mistake_trajectories[jnp.arange(final_mistake_trajectories.shape[0]),jnp.clip((relevant_ep_lengths-50),a_min=0).astype(int)]
            final_mistake_trajectories_starts = final_mistake_trajectories[jnp.arange(final_mistake_trajectories.shape[0]),jnp.clip((relevant_ep_lengths-10000),a_min=0).astype(int)]
            #indices = jnp.logical_and(safe_q_values_relevant<=90,safe_q_values_relevant>=35)
            #final_mistake_trajectories = final_mistake_trajectories_relevant[indices].reshape(-1,4)
            final_mistake_trajectories_starts = final_mistake_trajectories_starts.reshape(-1,4)
            final_mistake_trajectories_ends = final_mistake_trajectories_ends.reshape(-1,4)
            final_mistake_trajectories = jnp.concatenate(jnp.array([final_mistake_trajectories_starts, final_mistake_trajectories_ends]), axis=0)
            #safe_q_values_relevant = safe_q_values_relevant[indices].reshape(-1)
            print(f"Length potential starting states{len(final_mistake_trajectories)}")

            if len(final_mistake_trajectories)!=0:
                def softmax(x):
                    exp_x = jnp.exp(x - jnp.max(x))  # Subtract max for numerical stability
                    return exp_x / exp_x.sum()
                if len(final_mistake_trajectories)>100:
                    random_indices = jax.random.choice(key, final_mistake_trajectories.shape[0], shape=(100,), replace=False)# p = softmax(100-safe_q_values_relevant))
                    final_mistake_trajectories= final_mistake_trajectories[random_indices]
            #elif len(final_mistake_trajectories_relevant)!=0:
                #random_indices = jax.random.choice(key, final_mistake_trajectories_relevant[0].shape[0], shape=(100,), replace=False)
                #final_mistake_trajectories= final_mistake_trajectories_relevant[0][random_indices]


            all_action_taken_performance = perf_evaluator_output.action_taken_performance
            all_action_taken_safety= perf_evaluator_output.action_taken_safety
            all_trajectories = perf_evaluator_output.trajectories
            result_best_local = perf_evaluator_output.episode_metrics["episode_return"].max()
            if perf_eval_best_reward <= result_best_local:
                argmax_eval_index = perf_evaluator_output.episode_metrics["episode_return"].argmax()
                maximal_trajectories.append(all_trajectories[0][argmax_eval_index].tolist())
                maximal_actions_taken_performance.append(all_action_taken_performance[0][argmax_eval_index].tolist())
                maximal_actions_taken_safety.append(all_action_taken_safety[0][argmax_eval_index].tolist())
                perf_eval_best_reward = result_best_local

            if perf_eval_length >=config_s.env.solved_return_threshold:
                break

            custom_extras_safety = {"mistake_trajectories": final_mistake_trajectories}
            print(f"Len Mistake:{len(final_mistake_trajectories)}")

            key, safe_learn, _, _,_, safe_evaluator = generate_safety_env_and_learn(config_s, key, custom_extras_safety)

            start_time = time.time()
            safe_learner_output = safe_learn(safe_learner_state)
            jax.block_until_ready(safe_learner_output)

            # Log the results of the training.
            log_training_metrics_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, safe_learner_output,logger)
            start_time = time.time()
            key, safe_trained_params, eval_keys = prepare_safe_evaluation(config_s, key, safe_learner_output)
            safe_evaluator_output = safe_evaluator(safe_trained_params, eval_keys)
            jax.block_until_ready(safe_evaluator_output)
            # Log the results of the evaluation.

            plot_border_decisions(safe_evaluator_output.episode_metrics)

            elapsed_time = time.time() - start_time
            episode_return = log_evaluation_metrics_safety_training(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, safe_evaluator_output,logger)
            print(f"Episode Return of Training safety starting from mistake starting states: {episode_return}")

            safe_learner_state = safe_learner_output.learner_state
            eval_step_safety+=1

        config_p.arch.evaluation_greedy = True
        key, _, _, _, perf_evaluator_greedy=generate_performance_learner_and_evaluator(config_s,key,config_p, safe_actor_network, safe_q_network, safe_learner_state)
        config_p.arch.evaluation_greedy = False
        start_time = time.time()
        # Evaluate greedily
        perf_evaluator_output = perf_evaluator_greedy(perf_trained_params, perf_eval_keys)
        jax.block_until_ready(perf_evaluator_output)
        log_evaluation_metrics_performance_agent_greedy(config_s,config_p, elapsed_time, eval_step_safety,eval_step_perf, perf_evaluator_output,logger)

        key, perf_learn, _, _, perf_evaluator=generate_performance_learner_and_evaluator(config_s,key,config_p, safe_actor_network, safe_q_network, safe_learner_state)
        
        print(f"Start {i} performance training in Double Learning procedure")
        start_time = time.time()
        perf_learner_output = perf_learn(perf_learner_state)
        jax.block_until_ready(perf_learner_output)

        elapsed_time = time.time() - start_time    
        log_training_metrics_performance_training(config_p,config_s, elapsed_time, eval_step_perf,eval_step_safety, perf_learner_output, logger)
        
        eval_step_perf+=1
        # Update runner state to continue training.
        perf_learner_state = perf_learner_output.learner_state
    # Stop the logger.
    #Safe the trajectories for later videos
    with open("experienced_trajectories.pkl", "wb") as file:
        pickle.dump(maximal_trajectories, file)
    with open("experienced_actions_perf.pkl", "wb") as file:
        pickle.dump(maximal_actions_taken_performance, file)
    with open("experienced_actions_safe.pkl", "wb") as file:
        pickle.dump(maximal_actions_taken_safety, file)
    uploadVideos()#f"AblationBonusDoubleLearningJaxCartpole_Bonus{bonus},_{i}_Videos")
    logger.stop()
    return eval_step_safety


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_double_learning2.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    for j in range(10):
        for i in range(6):
            if j <= 5 or j>6:
                continue
            
            bonus = [0.1,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8][j]
            cfg_performance = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg_safety = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg_performance.arch.seed = cfg_safety.arch.seed + (i+2) * 36
            cfg_safety.arch.seed = cfg_safety.arch.seed + (i+2) * 36
            
            cfg_performance.system = list(cfg_safety.system.items())[1][1]
            cfg_safety.system = list(cfg_safety.system.items())[0][1]

            cfg_performance.network = list(cfg_safety.network.items())[1][1]
            cfg_safety.network = list(cfg_safety.network.items())[0][1]

            #print(list(cfg_safety.env.items()))
            cfg_performance.env = list(cfg_safety.env.items())[1][1]
            cfg_safety.env = list(cfg_safety.env.items())[0][1]

            #print("CFG Safety")
            #print(cfg_safety)
            #print("CFG Perf")
            #print(cfg_performance)
            # Run experiment.
            #jax.debug.breakpoint()
            cfg_safety.env.kwargs.bonus = bonus
            cfg_safety.logger.kwargs.name =f"AblationRangeOfVValuesWithFilterFactor_{i}"
            eval_performance = run_experiment(cfg_safety, cfg_performance)
            print(f"It took {eval_performance} Iterations to learn safety")
            print(f"{Fore.CYAN}{Style.BRIGHT}Double Learning experiment completed{Style.RESET_ALL}")


    return eval_performance


if __name__ == "__main__":
    print(jax.default_backend())
    hydra_entry_point()
