"""
Predictive Predator Tracker
Author: Ben McEwen

The class predator tracker takes mp4 file input (standard extract.py
format).

The centroid of the animal is determined (find_center()) and a Kalman
filter is used to smooth the movement and reduce noise introduced by
occlusion (kalman_filter()). A moving average filter and Kalman filter
is used to find the average velocity and this is used to calculate the
future position of the animal at a given time (predict_location()).
"""


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from kalman.main import main


class PredatorTracker:
    """Predator tracker class: - Finds centroid of animal
    - Uses kalman filter to smooth effects of occlusion
    - Determines direction vector of animal
    - Predicts future position of animal"""

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)

    def __init__(self, filename):
        self.filename = filename

    def find_center(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # convert the grayscale image to binary image
        _, thresh = cv2.threshold(gray_image, 127, 255, 0)

        # calculate moments of binary image
        M = cv2.moments(thresh)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        center = np.array([int(cX), int(cY)])

        return center, frame

    def find_direction(self, frame, center, previous_center):
        # Draw vector representing animals direction
        length = 1000

        # Calculate angle of line using current and previous point
        diff = center - previous_center
        if diff[0] != 0:
            angle = math.degrees(math.atan(diff[1] / diff[0]))
            print("Angle: ", angle)
        else:
            angle = 0
            print("Division by zero, angle set to 0")

        # Project point in front of animal
        diff[0] = int(length * (math.cos(math.radians(angle))))
        diff[1] = int(length * (math.sin(math.radians(angle))))
        point = center - diff

        # Plot line between current center and projected point
        cv2.line(frame, tuple(point), tuple(center), (0, 0, 0), 2)

        return angle

    def plot_results(self, real_x, real_y, estimated_x, estimated_y):
        # plot Animals real location and estimated location
        (real,) = plt.plot(real_x[3:], real_y[3:], "b-", label="Line 1")
        (estimate,) = plt.plot(estimated_x[3:], estimated_y[3:], "r-", label="Line 2")
        plt.xlim(0, 500)
        plt.ylim(360, 0)
        plt.title("Real vs Estimated Position")
        plt.xlabel("Pixel x")
        plt.ylabel("Pixel y")
        plt.legend([real, estimate], ["Real", "Estimate"])
        plt.grid(True)

        plt.savefig("results.png")

    def plot_prediction_results(
        self, estimated_x, estimated_y, predicted_x, predicted_y
    ):
        # plot Animals real location and estimated location
        (estimate,) = plt.plot(estimated_x[5:], estimated_y[5:], "b-", label="Line 1")
        (prediction,) = plt.plot(predicted_x[5:], predicted_y[5:], "r-", label="Line 2")
        plt.xlim(0, 500)
        plt.ylim(360, 0)
        plt.title("Real vs Estimated Position")
        plt.xlabel("Pixel x")
        plt.ylabel("Pixel y")
        plt.legend([estimate, prediction], ["Estimate", "Prediction"])
        plt.grid(True)

        plt.savefig("Prediction.png")

    def predict_location(self, frame, center, angle, velocity):
        # Return and plot the future predicted location
        offset = np.array([0, 0])
        distance = np.array([0, 0])

        # determine offset distance
        t = 4
        distance = velocity * t
        # print('distance: ', distance)
        length = np.linalg.norm(distance)
        # print('length: ', length)

        # Find location coordinate
        offset[0] = int(length * (math.cos(math.radians(angle))))
        offset[1] = int(length * (math.sin(math.radians(angle))))
        location = center - offset
        # print('Future location: ', location)

        # Plot line between current center and projected point
        cv2.circle(frame, tuple(location), 5, self.RED, thickness=-1)
        cv2.circle(frame, tuple(location), 10, self.RED, thickness=2)

        return location

    def plot_position(self, frame, center, estimated_center, predicted_center):
        # Initialise text function
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontsize = 0.5
        font_thickness = 1

        # Plot real, estimated and predicted position of animal
        # cv2.circle(frame, tuple(center), 5, self.RED, -1)
        cv2.circle(frame, tuple(estimated_center), 5, self.BLUE, -1)
        # cv2.circle(frame, tuple(predicted_center), 5, self.GREEN, -1)

        # cv2.putText(frame,'Predicted', (predicted_center[0]+10, predicted_center[1]), font, fontsize, self.GREEN, font_thickness, cv2.LINE_AA)
        # cv2.putText(frame,'Center', (center[0]-60, center[1]), font, fontsize, self.RED, font_thickness, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Estimated",
            (int(estimated_center[0] - 30), int(estimated_center[1] + 20)),
            font,
            fontsize,
            self.BLUE,
            font_thickness,
            cv2.LINE_AA,
        )

    def kalman_filter(self):
        cap = cv2.VideoCapture(self.filename)

        fourcc = cv2.VideoWriter_fourcc(*"X264")
        out = cv2.VideoWriter("output.avc1", fourcc, 20.0, (960, 720))

        timestep = 1 / 9

        count = 0

        initial_time = time.time()
        current_time = initial_time
        previous_time = 0
        elapsed_time = 0

        # Initialise arrays
        center = np.array([[0, 0]])

        previous_estimated = np.array([0, 0])
        estimated_state = np.array([0, 0])

        real_x = np.array([[0]])
        real_y = np.array([[0]])
        predicted_x = np.array([[0]])
        predicted_y = np.array([[0]])
        estimated_x = np.array([[0]])
        estimated_y = np.array([[0]])

        velocity = np.array([[0, 0], [0, 0]])

        velocity_x = 0
        velocity_y = 0

        subset_len = 2

        # Construct the 2-dimensional Kalman Filter and initialize the variables.
        kalman = cv2.KalmanFilter(4, 2, 4)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.controlMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.measurementMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], np.float32
        )
        kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.measurementNoiseCov = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.statePost = np.array([0, 0, 0, 0], np.float32)
        kalman.errorCovPost = np.array(
            [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]], np.float32
        )

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            cropped = frame[380:700, 0:450]

            original_frame = frame[20:360, 0:450]

            center, frame = self.find_center(cropped)

            # Calculate the predicted position of animal
            predicted_state = kalman.predict()
            print("predicted_state: ", predicted_state)
            predicted_center = np.array(
                [int(predicted_state[0]), int(predicted_state[1])]
            )

            # measured = np.array([[center[0]], [center[1]], [center[0]], [center[1]]], dtype="float32")
            measured = np.array(
                [[center[0]], [center[1]], [velocity_x], [velocity_y]], dtype="float32"
            )

            # Calculate the estimated position of animal
            estimated_state = kalman.correct(measured)
            estimated_center = np.array(
                [int(estimated_state[0]), int(estimated_state[1])]
            )
            estimated_velocity = np.array(
                [[int(estimated_state[2]), int(estimated_state[3])]]
            )
            angle = self.find_direction(frame, estimated_center, previous_estimated)

            # Predict future position using moving average of velocity
            velocity_x = (estimated_center[0] - previous_estimated[0]) / timestep
            velocity_y = (estimated_center[1] - previous_estimated[1]) / timestep

            if ((velocity_x != 0) and (velocity_y != 0)) and (
                (abs(velocity_x) < 100) and (abs(velocity_y) < 100)
            ):
                velocity = np.concatenate((velocity, estimated_velocity), axis=0)
            velocity = velocity[-subset_len:]

            velocity_sum = np.sum(velocity, axis=0)
            velocity_av = velocity_sum / subset_len

            print("velocity: ", velocity)
            print("velocity current:", velocity_x, velocity_y)
            print("velocity_av: ", velocity_av)

            predicted_loc = self.predict_location(
                frame, estimated_center, angle, velocity_av
            )
            # predicted_loc = self.predict_location(frame, estimated_center, angle, estimated_velocity)
            print("predicted_loc: ", predicted_loc)

            cv2.circle(original_frame, tuple(predicted_loc), 5, self.RED, thickness=-1)
            cv2.circle(original_frame, tuple(predicted_loc), 10, self.RED, thickness=2)

            # Plot real, estimated and predicted position of animal
            self.plot_position(frame, center, estimated_center, predicted_center)

            cv2.imshow("frame", frame)
            cv2.imshow("original_frame", original_frame)
            out.write(frame)

            # Remove 0,0 coordinates when animal is out of view
            if (center[0] != 0) or center[1] != 0:
                real_x = np.append((real_x), center[0])
                real_y = np.append((real_y), center[1])
                estimated_x = np.append((estimated_x), estimated_state[0])
                estimated_y = np.append((estimated_y), estimated_state[1])
                predicted_x = np.append((predicted_x), predicted_loc[0])
                predicted_y = np.append((predicted_y), predicted_loc[1])

            current_time = time.time()
            elapsed_time = elapsed_time + (current_time - previous_time)
            print("elapsed_time", elapsed_time)
            previous_time = current_time
            count = count + 1

            # Plot Animals actual location and estimated location
            # self.plot_results(real_x, real_y, estimated_x, estimated_y)
            self.plot_prediction_results(
                estimated_x, estimated_y, predicted_x, predicted_y
            )

            previous_estimated = estimated_center

            # Close the script if q is pressed.
            if cv2.waitKey(50) & 0xFF == ord("q"):

                # Calculate elapsed time
                average_elapsed_time = (elapsed_time - initial_time) / count
                print("average_elapsed_time", average_elapsed_time)

                # Calculate error accounting for estimate/prediction offset
                offset = 1

                sumX = sum(estimated_x[offset:])
                sumY = sum(estimated_y[offset:])
                length = len(estimated_x[offset:])

                predictionSumX = sum(predicted_x[1:])
                predictionSumY = sum(predicted_y[1:])

                errorX = abs(sumX - predictionSumX) / length
                errorY = abs(sumY - predictionSumY) / length
                print("errorX", errorX)
                print("errorY", errorY)
                break

        # Release the video file, and close the GUI.
        cap.release()
        out.release()
        cv2.destroyAllWindows()


main()
