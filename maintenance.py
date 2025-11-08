import numpy as np
from collections import deque
from config import (
    CURRENT_BASE, CURRENT_LOAD_FACTOR, MOTOR_NOISE,
    ROLLING_WINDOW, Z_ALERT
)


class MaintenanceMonitor:
    def __init__(self):
        self.window = deque(maxlen=ROLLING_WINDOW)
    def motor_current_model(self, v, w):
        return CURRENT_BASE + CURRENT_LOAD_FACTOR*(abs(v) + 0.7*abs(w)) + np.random.randn()*MOTOR_NOISE
    def zscore(self):
        if len(self.window) < 10:
            return 0.0
        arr = np.array(self.window, dtype=float)
        mu = arr.mean(); sd = arr.std() + 1e-6
        return (arr[-1] - mu)/sd
    def alert(self):
        return 1 if self.zscore() > Z_ALERT else 0
