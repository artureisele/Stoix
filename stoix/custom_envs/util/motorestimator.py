import numpy as np
import math

class MotorEstimator:
    def __init__(self):
        self.nx = 3
        self.nu = 1
        self.ny = 1
        self.dT = 1e-3

        self.A = np.array([[1, self.dT, self.dT**2 / 2],
                           [0, 1, self.dT],
                           [0, 0, 1]], dtype=float)

        self.K = np.array([9.91297949e-01, 1.14737440e+02, 6.59623054e+03], dtype=float)
        self.x = np.array([0, 0, 0], dtype=float)
        self.y_last = None
        self.revs = 0.0

    def predict(self):
        self.x = np.dot(self.A, self.x)

    def update(self, y):
        if self.y_last is None:
            self.y_last = y
        y_delta = y - self.y_last
        if abs(y_delta) > 2 * math.pi:
            return
        self.y_last = y
        while y_delta < -math.pi:
            y_delta += 2 * math.pi
        while y_delta > math.pi:
            y_delta -= 2 * math.pi
        self.revs += y_delta
        self.x = self.x + self.K * (self.revs - self.x[0])

    def get_propagated_est_angle(self, delay):
        return self.x[0] + delay * self.x[1]

    def get_propagated_meas_angle(self, delay):
        return self.revs + delay * self.x[1]

    def get_meas_angle(self):
        return self.revs

    def get_velocity(self):
        return self.x[1]

    def get_propagated_velocity(self, delay):
        return self.x[1] + delay * self.x[2]

    def get_acceleration(self):
        return self.x[2]
    
if __name__=="__main__":
    motor_estimator = MotorEstimator()
    for i in range(1000):
        motor_estimator.predict()
        motor_estimator.update(0)
        print(motor_estimator.get_meas_angle())
        print(motor_estimator.get_velocity())
        print(motor_estimator.get_acceleration())
