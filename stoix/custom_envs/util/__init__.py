from .estimator import Estimator
from .motorestimator import MotorEstimator
import numpy as np

def estimate(obs, estimator, motor_estimators):
    """
    Estimate the state of the robot using the provided estimator and motor estimators.
    """
    a_B = obs[:12].reshape(3, 4)
    omega_B = obs[12:24].reshape(3, 4)
    motor_angles = obs[24:]
    motor_states = np.zeros((3, 2))
    for i in range(2):
        motor_estimators[i].predict()
        motor_estimators[i].update(motor_angles[i])
        motor_states[0, i] = motor_estimators[i].get_meas_angle()
        motor_states[1, i] = motor_estimators[i].get_velocity()
        motor_states[2, i] = motor_estimators[i].get_acceleration()
    return estimator.update(omega_B, a_B, motor_states)