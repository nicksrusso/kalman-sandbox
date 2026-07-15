import numpy as np


class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, p0):
        self.F = F  # state transition matrix
        self.H = H  # observation matrix
        self.Q = Q  # process covariance
        self.R = R  # measurement covariance

        # Running belief states
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(p0, dtype=float)

    def predict(self):
        """Use the state transitions to take one timestep"""
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.matmul(self.H, self.x)  # surprise - how far off expected measurement actual is
        S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R  # how much of surprise is expected
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(S))  # prediction uncertainty / total uncertainty
        self.x = self.x + np.matmul(K, y)  # weight state update based on relative uncertainties
        self.P = np.matmul((np.eye(4) - np.matmul(K, self.H)), self.P)

        # Expose the innovation and its covariance for diagnostics (NIS).
        self.y = y
        self.S = S
