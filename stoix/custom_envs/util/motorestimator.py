import numpy as np
import math
import jax 
import jax.numpy as jnp
from flax import struct
import chex
class MotorEstimator:
    def __init__(self):
        self.nx = 3
        self.nu = 1
        self.ny = 1
        self.dT = 1e-3

        self.A = np.array([[1, self.dT, self.dT**2 / 2],
                           [0, 1, self.dT],
                           [0, 0, 1]], dtype=np.float32)

        self.K = np.array([9.91297949e-01, 1.14737440e+02, 6.59623054e+03], dtype=np.float32)
        self.x = np.array([0, 0, 0], dtype=np.float32)
        self.y_last = None
        self.revs = np.array(0.0, dtype=np.float32)

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
    
    def reset(self):
        self.x = np.array([0, 0, 0], dtype=np.float32)
        self.y_last = None
        self.revs = np.array(0.0, dtype=np.float32)
@struct.dataclass
class MotorEstimatorState():
    x: chex.Array
    y_last: chex.Numeric
    revs: chex.Numeric
class MotorEstimatorJax:
    def __init__(self):
        self.nx = 3
        self.nu = 1
        self.ny = 1
        self.dT = 1e-3

        self.A = jnp.array([[1, self.dT, self.dT**2 / 2],
                           [0, 1, self.dT],
                           [0, 0, 1]], dtype=float)

        self.K = jnp.array([9.91297949e-01, 1.14737440e+02, 6.59623054e+03], dtype=float)

    def reset(self)-> MotorEstimatorState:
        initial_state = MotorEstimatorState(
            x = jnp.array([0, 0, 0], dtype=float),
            y_last = jnp.finfo(jnp.float32).min,
            revs=0.0
        )
        return initial_state


    def predict(self, state_old: MotorEstimatorState):
        #self.x
        predictedState = MotorEstimatorState(
            x = jnp.dot(self.A, state_old.x),
            y_last = state_old.y_last,
            revs=state_old.revs
        )
        return predictedState

    def update(self, state_old: MotorEstimatorState, y_new)->MotorEstimatorState:
        y_last = state_old.y_last
        y_last = jax.lax.cond(y_last == jnp.finfo(jnp.float32).min, lambda: y_new, lambda:y_last)
        y_delta = y_new - y_last
        def wrap_to_pi(theta):
            return (theta + jnp.pi) % (2 * jnp.pi-0.000000000000001) - jnp.pi
        y_delta = wrap_to_pi(y_delta)
        new_revs = state_old.revs + y_delta

        new_x = state_old.x + self.K * (new_revs-state_old.x[0])
        new_state = MotorEstimatorState(
            x = new_x,
            y_last= y_new,
            revs= new_revs
        )
        return jax.lax.cond(jnp.abs(y_delta) > 2 * jnp.pi, lambda: state_old, lambda: new_state)

    def get_propagated_est_angle(self, state: MotorEstimatorState, delay):
        return state.x[0] + delay * state.x[1]

    def get_propagated_meas_angle(self, state: MotorEstimatorState, delay):
        return state.revs + delay * state.x[1]

    def get_meas_angle(self, state: MotorEstimatorState):
        return  state.revs

    def get_velocity(self, state: MotorEstimatorState):
        return state.x[1]

    def get_propagated_velocity(self, state: MotorEstimatorState, delay):
        return state.x[1] + delay * state.x[2]

    def get_acceleration(self, state: MotorEstimatorState):
        return state.x[2]
    

if __name__=="__main__":
    """
    They show roughly the same values
    motor_estimator = MotorEstimator()
    motor_estimator_jax = MotorEstimatorJax()
    state = motor_estimator_jax.reset()
    for i in range(100):
        motor_estimator.predict()
        state = motor_estimator_jax.predict(state)
        motor_estimator.update(0.1+0.001*i)
        state = motor_estimator_jax.update(state, 0.1+0.001*i)
        print("---")
        print(motor_estimator.get_meas_angle())
        print(motor_estimator_jax.get_meas_angle(state))
        print("---")
        print(motor_estimator.get_velocity())
        print(motor_estimator_jax.get_velocity(state))
        print("---")
        print(motor_estimator.get_acceleration())
        print(motor_estimator_jax.get_acceleration(state))
    """
    #Now in Jitted
    num_steps = 100
    inputs = 0.1 + 0.001 * jnp.arange(num_steps)
    # Define the scan function (replaces loop)
    motor_estimator_jax = MotorEstimatorJax()
    def scan_fn(state, measurement):
        state = motor_estimator_jax.predict(state)
        state = motor_estimator_jax.update(state, measurement)
        return state, state  # Return updated state and store it in output history
    state_init = motor_estimator_jax.reset()
    # Run scan (JIT-compilable, efficient iteration)
    scan_fn = jax.jit(scan_fn)
    final_state, state_history = jax.lax.scan(scan_fn, state_init, inputs)
    meas_angles = jax.jit(jax.vmap(motor_estimator_jax.get_meas_angle))(state_history)
    velocities = jax.jit(jax.vmap(motor_estimator_jax.get_velocity))(state_history)
    accelerations = jax.jit(jax.vmap(motor_estimator_jax.get_acceleration))(state_history)