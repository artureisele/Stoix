import numpy as np

def Rx(theta):
    """
    Rotation matrix around x-axis.
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def Ry(theta):
    """
    Rotation matrix around y-axis.
    """
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    """
    Rotation matrix around z-axis.
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def skew_symmetric(w):
    """
    Skew symmetric matrix.
    """
    v = w.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

class Estimator:
    def __init__(self, N_IMUS=4, N_MOTORS=2):
        self.N_IMUS = N_IMUS
        self.N_MOTORS = N_MOTORS

        self.R_upside_down = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.R01 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.R23 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.R_Bi = [self.R01, self.R01, self.R23, self.R23]
        # self.X1 = np.array([-0.166896, -0.167463, 0.667201, 0.667159]) # need to check this
        r = np.array([[-0.0234, -0.0204, 0.01939],
                        [0.0234, 0.0204, 0.01939],
                        [-0.016427, 0.020172, -0.01889],
                        [0.016368, -0.020481, -0.018916]]).T + np.array([[0, 0, 0.0325]]).T
        P = np.concatenate([np.ones((1,4)), r,])

        self.X = (P.T @ np.linalg.inv(P @ P.T))
        self.X1 = self.X[:,0]
        print("X1:", self.X1)

        self.pivot_accel = np.zeros(3)
        self.q_G = np.zeros(3)
        self.dq_G = np.zeros(3)
        self.q_A = np.zeros(2)
        self.q = np.zeros(3)
        self.dq = np.zeros(3)
        self.ddq = np.zeros(3)
        self.q_WR = np.zeros(N_MOTORS)
        self.dq_WR = np.zeros(N_MOTORS)
        self.ddq_WR = np.zeros(N_MOTORS)

        self.dt = 1.e-3
        self.r = 32e-3
        self.alpha = 0.02
        self.init = True

    def update(self, omega_B, a_B, motor_states):
        if self.init:
            g_B = self.calculate_g_B(a_B, self.pivot_accel, self.X1)
            self.q_A = self.estimate_accel(g_B)
            self.q[:2] = self.q_A
            self.q[2] = 0
            self.init = False

        m_q_WR = motor_states[0, :]
        m_dq_WR = motor_states[1, :]
        m_ddq_WR = motor_states[2, :]

        self.ddq_WR = m_ddq_WR
        self.dq_WR = m_dq_WR
        self.q_WR = m_q_WR

        self.dq_G = self.estimate_gyro(omega_B, self.q)
        self.q_G = self.integrate(self.dq_G, self.q, self.dq, self.dt)

        self.ddq = self.IIRFilter((self.dq_G - self.dq) / self.dt, self.ddq, 0.1)

        self.pivot_accel = self.estimate_pivot_accel(self.q_G, self.dq_G, self.ddq, self.dq_WR[0], self.ddq_WR[0])
        g_B = self.calculate_g_B(a_B, self.pivot_accel, self.X1)
        self.q_A = self.estimate_accel(g_B)

        alpha_scale = self.alpha 
        self.q[0] = alpha_scale * self.q_A[0] + (1 - alpha_scale) * self.q_G[0]
        self.q[1] = alpha_scale * self.q_A[1] + (1 - alpha_scale) * self.q_G[1]
        self.q[2] = self.q_G[2]

        self.dq = self.dq_G

        return np.array([self.q[2], self.q[0], self.q[1], self.dq[2], self.dq[0], self.dq[1], self.q_WR[0], self.dq_WR[0], self.q_WR[1], self.dq_WR[1]])

    def R1(self, q1):
        return np.array([[1, 0, 0], [0, np.cos(q1), -np.sin(q1)], [0, np.sin(q1), np.cos(q1)]])

    def R2(self, q2):
        return np.array([[np.cos(q2), 0, np.sin(q2)], [0, 1, 0], [-np.sin(q2), 0, np.cos(q2)]])

    def R3(self, q3):
        return np.array([[np.cos(q3), -np.sin(q3), 0], [np.sin(q3), np.cos(q3), 0], [0, 0, 1]])

    def jacobian_w2euler(self, q1, q2):
        return np.array([[np.cos(q2), 0, np.sin(q2)], [np.sin(q2) * np.tan(q1), 1, -np.cos(q2) * np.tan(q1)], [-np.sin(q2) / np.cos(q1), 0, np.cos(q2) / np.cos(q1)]])

    def average_vecs(self, m):
        return np.mean(m, axis=1)

    def estimate_gyro(self, w_B, q):
        J = self.jacobian_w2euler(q[0], q[1])
        w_avg = self.average_vecs(w_B)
        return J @ w_avg

    def integrate(self, dq, past_q, past_dq, dt):
        return (dq + past_dq) / 2.0 * dt + past_q

    def calculate_g_B(self, m_B, pivot_acc, X1):
        M = m_B - pivot_acc[:, np.newaxis]
        return M @ X1

    def estimate_accel(self, g_B):
        q_A = np.zeros(2)
        q_A[0] = np.arctan(g_B[1] / np.sqrt(g_B[0] ** 2 + g_B[2] ** 2))
        q_A[1] = -np.arctan(g_B[0] / g_B[2])
        return q_A

    def estimate_pivot_accel(self, q, dq, ddq, dq4, ddq4):
        c1 = np.cos(q[0])
        s1 = np.sin(q[0])

        temp1 = np.array([2 * c1 * self.r * dq[0] * dq[2] + self.r * s1 * ddq[2], -c1 * self.r * ddq[0] + self.r * s1 * dq[0] ** 2 + self.r * s1 * dq[2] ** 2, -c1 * self.r * dq[0] ** 2 - self.r * s1 * ddq[0]])
        ddp_WC = self.R2(q[1]).T @ self.R1(q[0]).T @ temp1

        temp2 = np.array([self.r * (ddq4+ddq[1]), self.r * dq[2] * (dq4+dq[1]), 0])
        ddp_CI = self.R2(q[1]).T @ self.R1(q[0]).T @ temp2

        return ddp_WC + ddp_CI
    
    def IIRFilter(self, new_value, old_value, alpha):
        return alpha * new_value + (1 - alpha) * old_value
    
    def reset(self):
        self.pivot_accel = np.zeros(3)
        self.q_G = np.zeros(3)
        self.dq_G = np.zeros(3)
        self.q_A = np.zeros(2)
        self.q = np.zeros(3)
        self.dq = np.zeros(3)
        self.ddq = np.zeros(3)
        self.q_WR = np.zeros(self.N_MOTORS)
        self.dq_WR = np.zeros(self.N_MOTORS)
        self.ddq_WR = np.zeros(self.N_MOTORS)

        self.dt = 1.e-3
        self.r = 32e-3
        self.alpha = 0.02
        self.init = True

if __name__=="__main__":
    estimator = Estimator()
    omega_B = np.zeros((3, 4))
    a_B = np.zeros((3, 4))
    a_B[2,:] = 9.81*np.cos(np.pi/6)
    a_B[0,:] = 9.81*np.sin(np.pi/6)
    motor_states = np.zeros((3, 2))
    result = estimator.update(omega_B, a_B, motor_states)
    print("omega_B:", omega_B)
    print("a_B:", a_B)
    print("motor_states:", motor_states)
    print("Update result:", result)
    print(np.pi/6)
