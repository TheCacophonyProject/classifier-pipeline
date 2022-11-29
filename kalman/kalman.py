import cv2
import numpy as np


class Kalman:
    def __init__(self):
        self.reset_kalman()

    def reset_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        self.kalman.processNoiseCov = np.eye(4, 4, dtype=np.float32) * 0.03

    def predict(self):
        return self.kalman.predict()

    def correct(self, rect):
        pts = np.array(
            [np.float32(rect.centroid[0]), np.float32(rect.centroid[1])], np.float32
        )
        self.kalman.correct(pts)
