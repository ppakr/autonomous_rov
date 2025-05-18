import numpy as np
from copy import deepcopy

class AlphaBetaFilter():
    def __init__(self, alpha=0.85, beta=0.005):
        self.alpha = alpha
        self.beta = beta
        self.position_estimate = None
        self.velocity_estimate = 0.0
        self.last_time = 0.0

    def filter(self, measurement, current_time):
        # measurement -> depth sensor data

        # if first time
        # position_est = position_measure
        # velocity_est = 0

        if self.position_estimate is None:
            # print("First time")
            self.position_estimate = measurement
            # print(self.position_estimate)
            self.last_time = deepcopy(current_time)
            # print(self.last_time)
            return self.position_estimate, self.velocity_estimate
        
        dt = current_time - self.last_time
        if dt <= 0:
            # print("dt <= 0")
            return self.position_estimate, self.velocity_estimate

        # Predict next position and velocity
        
        position_est_1 = self.position_estimate + dt * self.velocity_estimate
        velocity_est_1 = self.velocity_estimate

        # Update the position and velocity estimates
        # position_hat = position_est_1 + alpha * (position_measure - position_est_1)
        # velocity_hat = velocity_est_1 + beta * ((position_measure - position_est_1) / dt)
        residual = measurement - position_est_1
        position_hat = position_est_1 + self.alpha * residual
        velocity_hat = velocity_est_1 + (self.beta * residual) / dt

        # update the values
        self.position_estimate = position_hat
        self.velocity_estimate = velocity_hat

        self.last_time = deepcopy(current_time)

        return position_hat, velocity_hat
    

# simulation
# if __name__ == "__main__":
#     from time import sleep
#     import matplotlib.pyplot as plt

#     duration = 10 # seconds
#     dt = 0.1 # seconds

#     times = np.arange(0, duration, dt)

#     # print("Helloooo!")
#     # print("times", times)

#     true_position = 0.5 * times
#     measurements = true_position + np.random.normal(0, 0.1, len(times))

#     filter = AlphaBetaFilter(alpha=0.85, beta=0.005)

#     filtered_positions = []
#     filtered_velocities = []

#     for t, z in zip(times, measurements):
#         pos, vel = filter.filter(z, t)
#         filtered_positions.append(pos)
#         filtered_velocities.append(vel)

#     print("filtered_positions", filtered_positions)
#     print("filtered_velocities", filtered_velocities)

#     # Plotting
#     plt.figure(figsize=(12, 6))

#     plt.subplot(2, 1, 1)
#     plt.plot(times, true_position, label='True Position', linewidth=2)
#     plt.plot(times, measurements, label='Noisy Measurements', alpha=0.5)
#     plt.plot(times, filtered_positions, label='Filtered Position', linewidth=2)
#     plt.ylabel('Position')
#     plt.title('Alpha-Beta Filter Position Estimation')
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(times, [0.5]*len(times), label='True Velocity', linestyle='--')
#     plt.plot(times, filtered_velocities, label='Estimated Velocity')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Velocity')
#     plt.title('Alpha-Beta Filter Velocity Estimation')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

