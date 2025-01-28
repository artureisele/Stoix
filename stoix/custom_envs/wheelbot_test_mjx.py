import numpy as np
from util import Estimator, MotorEstimator, estimate
import os
from typing import Dict, Tuple, Union, Optional, List, Sequence
import mujoco
from mujoco import mjx
from gymnax.environments import spaces
import jax.numpy as jnp
import jax 
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.0)),
    "elevation": -90.0,
}

DEFAULT_SIZE = 480
class WheelbotEnv():
    def __init__(
        self,
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_function = None,
        estimate_states = False,
        **kwargs,
    ):  
        self.frame_skip = frame_skip
        self.opt_timestep = 0.001
        xml_file = os.path.join(os.path.dirname(__file__), 'mjcf', 'wheelbot_alpha.xml')
        xml_file = os.path.abspath(xml_file)
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = self.opt_timestep
        self.model.vis.global_.offwidth = DEFAULT_SIZE
        self.model.vis.global_.offheight = DEFAULT_SIZE
        self.data = mujoco.MjData(self.model)
        #mujoco.mj_forward(model, data)

        #self.imu_rotation_matrices = {
        #                            'imu_1': self.data.site_xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'imu_1')].reshape(3, 3).copy(),
        #                            'imu_2': self.data.site_xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'imu_2')].reshape(3, 3).copy(),
        #                            'imu_3': self.data.site_xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'imu_3')].reshape(3, 3).copy(),
        #                            'imu_4': self.data.site_xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'imu_4')].reshape(3, 3).copy()
        #                        }
        
        self.metadata = {
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.estimate_states = estimate_states
        self.estimator = Estimator()
        self.motorestimators = [MotorEstimator(), MotorEstimator()]
        if self.estimate_states:
            obs_size = 10
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
            self.observation_structure = {
                "yaw": 1,
                "roll": 1,
                "pitch": 1,
                "yaw_rate": 1,
                "roll_rate": 1,
                "pitch_rate": 1,
                "drive_wheel_angle": 1,
                "drive_wheel_angle_velocity": 1,
                "reaction_wheel_angle": 1,
                "reaction_wheel_angle_velocity": 1,
            }
        else:
            obs_size = 26

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
            )

            self.observation_structure = {
                "acc_x": 4,
                "acc_y": 4,
                "acc_z": 4,
                "gyr_x": 4,
                "gyr_y": 4,
                "gyr_z": 4,
                "motor_angles": 2,
            }
        self.reward_function = reward_function
    def init_mjx(self):
        mjx_model = mjx.put_model(self.model)
        mjx_data = mjx.put_data(self.model, self.data)
        return mjx_model, mjx_data
    @property
    def action_space(self) -> spaces.Box:
        """Action space of the environment."""
        high_action = jnp.array(0.1)
        return spaces.Box(-high_action, high_action, (2,), dtype=jnp.float32)
    @property
    def dt(self) -> float:
        return self.opt_timestep * self.frame_skip
    
    def _get_obs(self, model, data):
      #sensor_start_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'acc_1')-> just use 0 for now
      sensor_start_idx = 0
      #a_B = np.array([
      #        self.imu_rotation_matrices['imu_1'] @ self.data.sensordata[sensor_start_idx:sensor_start_idx+3],
      #        self.imu_rotation_matrices['imu_2'] @ self.data.sensordata[sensor_start_idx+3:sensor_start_idx+6],
      #        self.imu_rotation_matrices['imu_3'] @ self.data.sensordata[sensor_start_idx+6:sensor_start_idx+9],
      #        self.imu_rotation_matrices['imu_4'] @ self.data.sensordata[sensor_start_idx+9:sensor_start_idx+12]
      #        ]).T
      a_B = jnp.array([
          data.site_xmat[0] @ data.sensordata[sensor_start_idx:sensor_start_idx+3],
          data.site_xmat[1] @ data.sensordata[sensor_start_idx+3:sensor_start_idx+6],
          data.site_xmat[2] @ data.sensordata[sensor_start_idx+6:sensor_start_idx+9],
          data.site_xmat[3] @ data.sensordata[sensor_start_idx+9:sensor_start_idx+12],
      ])    
      """         
        omega_B = np.array([
            self.imu_rotation_matrices['imu_1'] @ self.data.sensordata[sensor_start_idx+12:sensor_start_idx+15],
            self.imu_rotation_matrices['imu_2'] @ self.data.sensordata[sensor_start_idx+15:sensor_start_idx+18],
            self.imu_rotation_matrices['imu_3'] @ self.data.sensordata[sensor_start_idx+18:sensor_start_idx+21],
            self.imu_rotation_matrices['imu_4'] @ self.data.sensordata[sensor_start_idx+21:sensor_start_idx+24]
        ]).T
      """

      omega_B = jnp.array([
          data.site_xmat[0] @ data.sensordata[sensor_start_idx+12:sensor_start_idx+15],
          data.site_xmat[1] @ data.sensordata[sensor_start_idx+15:sensor_start_idx+18],
          data.site_xmat[2] @ data.sensordata[sensor_start_idx+18:sensor_start_idx+21],
          data.site_xmat[3] @ data.sensordata[sensor_start_idx+21:sensor_start_idx+24],
      ])    

      motor_angles = jnp.array([
          data.sensordata[sensor_start_idx+24],
          data.sensordata[sensor_start_idx+25]
      ])
      def trueFun():
          return jnp.concatenate((a_B.flatten(), omega_B.flatten(), motor_angles))
      def falseFun():
        #return jax.lax.cond(arg1, lambda:estimate(np.concatenate((a_B.flatten(), omega_B.flatten(), motor_angles)), arg2, arg3), lambda: np.concatenate((a_B.flatten(), omega_B.flatten(), motor_angles)))
        return jnp.concatenate((a_B.flatten(), omega_B.flatten(), motor_angles))
      return jax.lax.cond( jnp.logical_or( jnp.logical_or((a_B==0).all() , (omega_B==0).all()),(motor_angles==0).all()),trueFun, falseFun, )#(self.estimate_states, self.estimator, self.motorestimators ))  # If all the accelerometer values are zero, repeat step


      return obs

    def _check_terminated(self):
        data = self.data
        model = self.model
        fall = False
        wheel_contact = False
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom1=="wheel_1_geom" and geom2=="ground") or (geom2=="wheel_1_geom" and geom1=="ground"):
                wheel_contact = True 
            if (geom1=="body_geom" and geom2=="ground") or (geom2=="body_geom" and geom1=="ground"):
                fall = True 
                break
        if fall and not wheel_contact:
            print("Is fallen!!!!")
            return True
        else:
            return False


    def step(self,model, data, action):
        data.replace(ctrl=action)
        new_data = mjx.step(model,data)

        observation = self._get_obs(model, data)
        #reward, reward_info = self._get_rew(observation, action)
        #terminated = self._check_terminated()
        info = {
            "qpos": data.qpos.flatten(),
            "qvel": data.qvel.flatten(),
            #**reward_info,
        }
        return new_data, observation, info

    def _get_rew(self, observation: float, action):
        if not self.reward_function:
            print("You have not specified a reward function! Use set_reward!")
            reward = 0
            reward_info = {}
        else:
            reward,reward_info = self.reward_function(observation, action)

        return reward, reward_info
    
    def set_reward(self, reward_function):
        self.reward_function = reward_function

    def reset_model(self):
        noise_scale_pos = 0  # Scale of noise for positions
        noise_scale_vel = 0  # Scale of noise for velocities
        noisy_qpos = self.init_qpos + np.random.uniform(
            low=-noise_scale_pos, high=noise_scale_pos, size=self.init_qpos.shape
        )
        noisy_qvel = self.init_qvel + np.random.uniform(
            low=-noise_scale_vel, high=noise_scale_vel, size=self.init_qvel.shape
        )
        self.set_state(noisy_qpos, noisy_qvel)
        if self.estimate_states:
            # Initialize the Estimator
            self.estimator = Estimator()
            self.motorestimators = [MotorEstimator(), MotorEstimator()]
            print("Reset estiamtors")

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "qpos": self.data.qpos.flatten(),
            "qvel": self.data.qvel.flatten(),
        }
    def render(
        self,
        model,
        trajectory: List[mjx.Data],
        camera: Optional[str] = None,
    ) -> Sequence[np.ndarray]:
      """Renders a trajectory using the MuJoCo renderer."""
      renderer = mujoco.Renderer(model, height=DEFAULT_SIZE, width=DEFAULT_SIZE)
      camera = camera or -1

      def get_image(state: mjx.Data):
        d = mujoco.MjData(model)
        d.qpos, d.qvel = state.qpos, state.qvel
        mujoco.mj_forward(model, d)
        renderer.update_scene(d, camera=camera)
        return renderer.render()

      if isinstance(trajectory, list):
        return [get_image(s) for s in trajectory]

      return get_image(trajectory)
