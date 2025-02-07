import numpy as np


class DecoupledRollPitchLqrController:
    YAW, ROLL, PITCH, YAW_RATE, ROLL_RATE, PITCH_RATE, DRIVING_WHEEL_ANGLE, DRIVING_WHEEL_ANGULAR_VELOCITY, BALANCING_WHEEL_ANGLE, BALANCING_WHEEL_ANGULAR_VELOCITY = range(10)
    TAU_DRIVING_WHEEL, TAU_BALANCING_WHEEL = range(2)

    roll_states = [ROLL, ROLL_RATE, BALANCING_WHEEL_ANGLE, BALANCING_WHEEL_ANGULAR_VELOCITY]
    roll_inputs = [TAU_BALANCING_WHEEL]

    pitch_states = [PITCH, PITCH_RATE, DRIVING_WHEEL_ANGLE, DRIVING_WHEEL_ANGULAR_VELOCITY]
    pitch_inputs = [TAU_DRIVING_WHEEL]

    wheel_roll_ang_windup = 1000
    wheel_pitch_ang_windup = 10000

    def __init__(self, K_roll, K_pitch, u_max=0.5, max_angles=np.inf):
        self.K_roll = K_roll
        self.K_pitch = K_pitch
        self.u_max = u_max
        self.max_angles = max_angles
        self.wheel_roll_ang = 0
        self.wheel_roll_ang_last = 0
        self.wheel_pitch_ang = 0
        self.wheel_pitch_ang_last = 0

    def update(self, x, setpoint=np.zeros(10)):
        u = np.zeros(2)
        if np.any(np.abs(x[[self.ROLL, self.PITCH]]) > self.max_angles): # If the robot is too tilted, stop the motors
            # print("Robot is too tilted")
            return u

        err_x = x - setpoint

        self.wheel_roll_ang += err_x[self.BALANCING_WHEEL_ANGLE] - self.wheel_roll_ang_last
        self.wheel_pitch_ang += err_x[self.DRIVING_WHEEL_ANGLE] - self.wheel_pitch_ang_last

        self.wheel_roll_ang_last = err_x[self.BALANCING_WHEEL_ANGLE]
        self.wheel_pitch_ang_last = err_x[self.DRIVING_WHEEL_ANGLE]

        self.wheel_roll_ang = self._windup(self.wheel_roll_ang, self.wheel_roll_ang_windup)
        self.wheel_pitch_ang = self._windup(self.wheel_pitch_ang, self.wheel_pitch_ang_windup)

        err_x[self.BALANCING_WHEEL_ANGLE] = self.wheel_roll_ang
        err_x[self.DRIVING_WHEEL_ANGLE] = self.wheel_pitch_ang

        u[self.roll_inputs] = -np.dot(self.K_roll, err_x[self.roll_states])
        u[self.pitch_inputs] = -np.dot(self.K_pitch, err_x[self.pitch_states])
        return np.clip(u, -self.u_max, self.u_max)

    def _windup(self, angle, windup):
        if angle > windup:
            return angle - windup
        elif angle < -windup:
            return angle + windup
        return angle
    
    def reset(self):
        self.wheel_roll_ang = 0
        self.wheel_roll_ang_last = 0
        self.wheel_pitch_ang = 0
        self.wheel_pitch_ang_last = 0

if __name__=="__main__":
    K_roll = np.array([1.3e0,  1.6e-1,  0.8e-04, 4e-04])
    K_pitch = np.array([600e-3, 40e-3, 4e-3,  3e-3])
    controller = DecoupledRollPitchLqrController(K_roll, K_pitch)
    x = np.zeros(10)
    x[controller.DRIVING_WHEEL_ANGLE] = 0.1
    u = controller.update(x)
    print(u)
