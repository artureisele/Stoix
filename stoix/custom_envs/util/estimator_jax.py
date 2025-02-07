import jax 
import jax.numpy as jnp
from flax import struct
import chex

def Rx(theta):
    """
    Rotation matrix around x-axis.
    """
    return jnp.array([[1, 0, 0],
                     [0, jnp.cos(theta), -jnp.sin(theta)],
                     [0, jnp.sin(theta), jnp.cos(theta)]])

def Ry(theta):
    """
    Rotation matrix around y-axis.
    """
    return jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                     [0, 1, 0],
                     [-jnp.sin(theta), 0, jnp.cos(theta)]])

def Rz(theta):
    """
    Rotation matrix around z-axis.
    """
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                     [jnp.sin(theta), jnp.cos(theta), 0],
                     [0, 0, 1]])


def skew_symmetric(w):
    """
    Skew symmetric matrix.
    """
    v = w.flatten()
    return jnp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

@struct.dataclass
class EstimatorState():
    pivot_accel: chex.Array
    q_G : chex.Array
    dq_G : chex.Array
    q : chex.Array
    ddq : chex.Array
    q_WR : chex.Array
    dq_WR : chex.Array
    ddq_WR : chex.Array
    init : chex.Numeric

class Estimator:
    def __init__(self, N_IMUS=4, N_MOTORS=2):
        self.N_IMUS = N_IMUS
        self.N_MOTORS = N_MOTORS

        self.R_upside_down = jnp.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.R01 = jnp.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.R23 = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.R_Bi = [self.R01, self.R01, self.R23, self.R23]
        # self.X1 = jnp.array([-0.166896, -0.167463, 0.667201, 0.667159]) # need to check this
        r = jnp.array([[-0.0234, -0.0204, 0.01939],
                        [0.0234, 0.0204, 0.01939],
                        [-0.016427, 0.020172, -0.01889],
                        [0.016368, -0.020481, -0.018916]]).T + jnp.array([[0, 0, 0.0325]]).T
        P = jnp.concat([jnp.ones((1,4)), r,])

        self.X = (P.T @ jnp.linalg.inv(P @ P.T))
        self.X1 = self.X[:,0]
        self.r = 32e-3
        self.dt = 1.e-3
        self.alpha = 0.02
        jax.debug.print("X1:{}", self.X1)


    def update(self, estimator_state_old:EstimatorState, omega_B, a_B, motor_states)->tuple[EstimatorState, chex.Array]:
        #Redundant computation after init, however easier to implement here for jax
        g_B = self.calculate_g_B(a_B, estimator_state_old.pivot_accel, self.X1)
        q_A_new_init = self.estimate_accel(g_B)
        q_new_init = jnp.concat([q_A_new_init, jnp.array([0])], axis = 0)
        init_updated_state = estimator_state_old.replace(q = q_new_init)
        init_updated_state = init_updated_state.replace(init = 0.0)
        #Now Regular update
        m_q_WR = motor_states[0, :]
        m_dq_WR = motor_states[1, :]
        m_ddq_WR = motor_states[2, :]

        dq_G_new = self.estimate_gyro(omega_B, estimator_state_old.q)
        q_G_new = self.integrate(dq_G_new, estimator_state_old.q, estimator_state_old.dq_G, self.dt)

        ddq_new = self.IIRFilter((dq_G_new - estimator_state_old.dq_G) / self.dt, estimator_state_old.ddq, 0.1)

        pivot_accel_new = self.estimate_pivot_accel(q_G_new, dq_G_new, ddq_new, m_dq_WR[0], m_ddq_WR[0])
        g_B_new = self.calculate_g_B(a_B, pivot_accel_new, self.X1)
        q_A_new = self.estimate_accel(g_B_new)

        alpha_scale = self.alpha

        
        q_0 = alpha_scale * q_A_new[0] + (1 - alpha_scale) * q_G_new[0]
        q_1 = alpha_scale * q_A_new[1] + (1 - alpha_scale) * q_G_new[1]
        q_2 = q_G_new[2]
        q_new = jnp.array([q_0,q_1,q_2]) 

        regular_updated_state = EstimatorState(
            pivot_accel = pivot_accel_new,
            q_G = q_G_new,
            dq_G = dq_G_new,
            q = q_new,
            ddq = ddq_new,
            q_WR = m_q_WR,
            dq_WR = m_dq_WR,
            ddq_WR = m_ddq_WR,
            init = estimator_state_old.init,
        )
        updated_state =  jax.lax.cond(estimator_state_old.init == 1.0, lambda: init_updated_state, lambda: regular_updated_state)
        return updated_state ,jnp.array([updated_state.q[2], updated_state.q[0], updated_state.q[1], updated_state.dq_G[2], updated_state.dq_G[0], updated_state.dq_G[1], updated_state.q_WR[0], updated_state.dq_WR[0], updated_state.q_WR[1], updated_state.dq_WR[1]])

    def R1(self, q1):
        return jnp.array([[1, 0, 0], [0, jnp.cos(q1), -jnp.sin(q1)], [0, jnp.sin(q1), jnp.cos(q1)]])

    def R2(self, q2):
        return jnp.array([[jnp.cos(q2), 0, jnp.sin(q2)], [0, 1, 0], [-jnp.sin(q2), 0, jnp.cos(q2)]])

    def R3(self, q3):
        return jnp.array([[jnp.cos(q3), -jnp.sin(q3), 0], [jnp.sin(q3), jnp.cos(q3), 0], [0, 0, 1]])

    def jacobian_w2euler(self, q1, q2):
        return jnp.array([[jnp.cos(q2), 0, jnp.sin(q2)], [jnp.sin(q2) * jnp.tan(q1), 1, -jnp.cos(q2) * jnp.tan(q1)], [-jnp.sin(q2) / jnp.cos(q1), 0, jnp.cos(q2) / jnp.cos(q1)]])

    def average_vecs(self, m):
        return jnp.mean(m, axis=1)

    def estimate_gyro(self, w_B, q):
        J = self.jacobian_w2euler(q[0], q[1])
        w_avg = self.average_vecs(w_B)
        return J @ w_avg

    def integrate(self, dq, past_q, past_dq, dt):
        return (dq + past_dq) / 2.0 * dt + past_q

    def calculate_g_B(self, m_B, pivot_acc, X1):
        M = m_B - pivot_acc[:, jnp.newaxis]
        return M @ X1

    def estimate_accel(self, g_B):
        q_A = jnp.zeros(2)
        q_A = q_A.at[0].set(jnp.arctan(g_B[1] / jnp.sqrt(g_B[0] ** 2 + g_B[2] ** 2)))
        q_A = q_A.at[1].set( -jnp.arctan(g_B[0] / g_B[2]))
        return q_A

    def estimate_pivot_accel(self, q, dq, ddq, dq4, ddq4):
        c1 = jnp.cos(q[0])
        s1 = jnp.sin(q[0])

        temp1 = jnp.array([2 * c1 * self.r * dq[0] * dq[2] + self.r * s1 * ddq[2], -c1 * self.r * ddq[0] + self.r * s1 * dq[0] ** 2 + self.r * s1 * dq[2] ** 2, -c1 * self.r * dq[0] ** 2 - self.r * s1 * ddq[0]])
        ddp_WC = self.R2(q[1]).T @ self.R1(q[0]).T @ temp1

        temp2 = jnp.array([self.r * (ddq4+ddq[1]), self.r * dq[2] * (dq4+dq[1]), 0])
        ddp_CI = self.R2(q[1]).T @ self.R1(q[0]).T @ temp2

        return ddp_WC + ddp_CI
    
    def IIRFilter(self, new_value, old_value, alpha):
        return alpha * new_value + (1 - alpha) * old_value
    
    def reset(self)->EstimatorState:
        initial_state = EstimatorState(
            pivot_accel = jnp.zeros(3),
            q_G = jnp.zeros(3),
            dq_G = jnp.zeros(3),
            q = jnp.zeros(3),
            ddq = jnp.zeros(3),
            q_WR = jnp.zeros(self.N_MOTORS),
            dq_WR = jnp.zeros(self.N_MOTORS),
            ddq_WR = jnp.zeros(self.N_MOTORS),
            init = 1.0,
        )
        return initial_state

if __name__=="__main__":
    estimator = Estimator()
    state = estimator.reset()
    omega_B = jnp.zeros((3, 4))
    a_B = jnp.zeros((3, 4))
    a_B = a_B.at[2,:].set(9.81*jnp.cos(jnp.pi/6))
    a_B = a_B.at[0,:].set(9.81*jnp.sin(jnp.pi/6))
    motor_states = jnp.zeros((3, 2))
    state ,result = estimator.update(state, omega_B, a_B, motor_states)

    num_steps = 100
    inputs = 0.1 + 0.001 * jnp.arange(num_steps)
    def scan_fn(state, _):
        state = state ,result = estimator.update(state, omega_B, a_B, motor_states)
        return state, result  # Return updated state and store it in output history
    state_init = state
    # Run scan (JIT-compilable, efficient iteration)
    scan_fn = jax.jit(scan_fn)
    final_state, state_history = jax.lax.scan(scan_fn, state_init, inputs)

    print("omega_B:", omega_B)
    print("a_B:", a_B)
    print("motor_states:", motor_states)
    print("Update result:", state_history)
    print(jnp.pi/6)
