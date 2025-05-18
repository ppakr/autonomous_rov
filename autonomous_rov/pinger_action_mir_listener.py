import rclpy
import traceback
import numpy as np
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from struct import pack, unpack
from std_msgs.msg import Int16, Float64, Empty, Float64MultiArray, String, Bool
from sensor_msgs.msg import Joy, Imu, FluidPressure, LaserScan
from mavros_msgs.srv import CommandLong, SetMode, StreamRate
from mavros_msgs.msg import OverrideRCIn, Mavlink
from mavros_msgs.srv import EndpointAdd
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
from autonomous_rov.PIDController import PIDController
# from autonomous_rov.CubicTrajectory import CubicTrajectory
from autonomous_rov.AlphaBetaFilter import AlphaBetaFilter
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult, FloatingPointRange
from rclpy.parameter import ParameterType

# Import action library and your custom action interface
import rclpy.action
from autonomous_rov.action import PingerAvoidance # Replace autonomous_rov with your package name if different
from rclpy.duration import Duration # Import Duration for sleeping

class MyPythonNode(Node):
    def __init__(self):
        super().__init__("listenerMIR")
        self.get_logger().info("This node is named listenerMIR")
        self.ns = self.get_namespace()
        self.get_logger().info("namespace =" + self.ns)

        # Publishers
        self.pub_msg_override = self.create_publisher(OverrideRCIn, "rc/override", 10)
        self.pub_angle_degree = self.create_publisher(Twist, 'angle_degree', 10)
        self.pub_depth_error = self.create_publisher(Float64, '/depth_error_val', 10)
        self.pub_angular_velocity = self.create_publisher(Twist, 'angular_velocity', 10)
        self.pub_depth_pwm = self.create_publisher(Float64, '/depth_pwm', 10)
        self.yaw_error_pwm = self.create_publisher(Float64, '/yaw_error_pwm', 10)
        self.pub_generated_traj = self.create_publisher(Pose, 'generated_traj', 10)
        self.pub_generated_traj_dot = self.create_publisher(Twist, 'generated_traj_dot', 10)

        self.get_logger().info("Publishers created.")

        # Action Server for Pinger Avoidance
        self._action_server = rclpy.action.ActionServer(
            self,
            PingerAvoidance,
            'pinger_avoidance',
            self.execute_callback,
            handle_goal=self.handle_goal,
            handle_cancel=self.handle_cancel)

        self.get_logger().info("PingerAvoidance Action Server created.")


        self.get_logger().info("ask router to create endpoint to enable mavlink/from publication.")

        self.armDisarm(False)  # Not automatically disarmed at startup
        rate = 25  # 25 Hz
        self.setStreamRate(rate)
        self.subscriber()

        self.control_rate = 20.0  # Hz
        self.control_period = 1.0 / self.control_rate
        time_tupple = self.get_clock().now().seconds_nanoseconds()
        self.time = time_tupple[0] + (time_tupple[1] * 10**-9)
        timer_period = 0.05  # 50 msec - 20 Hz
        self.timer = self.create_timer(self.control_period, self.timer_callback)
        self.set_mode = [0] * 4
        self.set_mode[0] = True  # Mode manual
        self.set_mode[1] = False  # Mode automatic without correction (cmd_vel)
        self.set_mode[2] = False  # Mode with correction (PID depth/yaw)
        self.set_mode[3] = False  # Mode with pinger (Action)
        self.init_a0 = True
        self.init_p0 = True
        self.arming = False

        # Initial states/values
        self.angle_roll_a0 = 0.0
        self.angle_pitch_a0 = 0.0
        self.angle_yaw_a0 = 0.0
        self.depth_wrt_startup = 0
        self.depth_p0 = 0

        # --- Pinger related variables (now managed by the action execution logic) ---
        self.latest_pinger_distance = 0.0
        self.latest_pinger_confidence = 0 # Store latest data from callback
        self.pinger_prev_error = 0.0      # Used in the action execution
        self.pinger_error_change = 0.0    # Used in the action execution
        self.pinger_error = 0.0           # Used in the action execution
        self.pinger_threshold = 0.75      # Used in the action execution (can be a parameter)
        self._pinger_avoidance_goal_handle = None # To keep track of the active goal handle
        self.current_pinger_state = "idle" # State for feedback
        self.free_path = False            # State flag for action logic
        self.search_path = True           # State flag for action logic
        self.crab_walk = False            # State flag (from subscription, used in action logic)
        self.positive_rotation = True     # State flag for search logic
        # --- End Pinger related variables ---


        self.Vmax_mot = 1900
        self.Vmin_mot = 1100

        # PWM commands - These will be updated by different modes/controllers
        self.Correction_yaw_pwm = 1500
        self.Correction_depth = 1500
        self.surge_pwm = 1500
        self.sway_pwm = 1500

        # controller parameters
        self.config = {}
        self.pid_depth = PIDController(type='linear')
        self.pid_yaw = PIDController(type='angular')
        self.pid_surge = PIDController(type='linear') # PID for surge in pinger mode
        self.pid_sway = PIDController(type='linear') # PID for sway in pinger mode (if needed)

        self.declare_and_set_params() # PID parameters usually
        self.add_on_set_parameters_callback(self.callback_params)

        self.desired_depth = 0.0
        self.desired_yaw = 0.0 # Target yaw for PID in corrected mode

        # self.depth_filter = AlphaBetaFilter(alpha=0.85, beta=0.005)
        self.yaw_filter = AlphaBetaFilter(alpha=0.85, beta=0.005)

        # Initialize trajectory but do not start
        # self.trajectory = CubicTrajectory(z_init=self.depth_p0, z_final=-0.2)
        self.traj_active = False  # Trajectory state
        self.time_init = None
        self.time_final = None
        # self.desired_depth = 0.0

        # Service to start trajectory (Depth Trajectory)
        self.srv = self.create_service(SetBool, 'start_trajectory', self.trajectory_callback)

    # --- Action Server Callbacks ---
    def handle_goal(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info(f'Received PingerAvoidance goal request: {goal_request}')
        # Accept the goal
        return rclpy.action.GoalResponse.ACCEPT

    def handle_cancel(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received PingerAvoidance cancel request')
        # Accept the cancel request
        return rclpy.action.CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute a PingerAvoidance goal."""
        self.get_logger().info('Executing PingerAvoidance goal...')

        # Store the goal handle to check for cancellation and send results/feedback
        self._pinger_avoidance_goal_handle = goal_handle

        feedback_msg = PingerAvoidance.Feedback()
        result = PingerAvoidance.Result()

        # Reset state flags for the action execution
        self.free_path = False
        self.search_path = True # Start in search mode
        self.positive_rotation = True # Start rotating one way in search

        # Initialize PWMs to neutral before starting movement (optional, but safe)
        self.surge_pwm = 1500
        self.sway_pwm = 1500
        # self.Correction_yaw_pwm = 1500 # Yaw is often handled by a separate PID even in this mode

        # Rate for the action execution loop
        rate = self.create_rate(self.control_rate) # Use the same control rate

        # Loop until goal is completed or cancelled
        while rclpy.ok():
            # Check for cancelation
            if goal_handle.is_cancel_requested:
                self.get_logger().info('PingerAvoidance goal canceled.')
                result.success = False
                result.message = "Pinger avoidance cancelled"
                # Reset PWMs to neutral on cancellation
                self.surge_pwm = 1500
                self.sway_pwm = 1500
                self.Correction_yaw_pwm = 1500 # Also reset yaw in this mode
                goal_handle.canceled()
                return result

            # --- Pinger Avoidance/Search Logic (Moved from pinger_callback) ---

            current_time = self.get_clock().now().seconds_nanoseconds()[0] + \
                           self.get_clock().now().seconds_nanoseconds()[1] * 1e-9

            pinger_distance = self.latest_pinger_distance
            pinger_confidence = self.latest_pinger_confidence
            current_yaw = self.get_yaw_degrees() # Need a way to get current yaw

            self.pinger_threshold = 1.0 # Can be a parameter

            if pinger_confidence < 60:
                self.current_pinger_state = "low_confidence"
                # Stop movement if confidence is too low
                self.surge_pwm = 1500
                self.sway_pwm = 1500
                # self.Correction_yaw_pwm = 1500 # Keep yaw control neutral if confidence is bad
                self.search_path = True # Assume we need to search if confidence drops
                self.free_path = False

            else: # Confidence is high enough
                if pinger_distance < self.pinger_threshold:
                    # Obstacle detected: closer than safe threshold
                    self.current_pinger_state = "avoiding"
                    self.free_path = False
                    self.search_path = True # Go into search mode to find a clear path

                    # Use PID for surge based on distance error
                    # Note: PID needs tuning for this specific application
                    surge_control_output = self.pid_surge.calculate_pid(
                        self.pinger_threshold, pinger_distance, current_time # Target is threshold, state is distance
                    )
                    self.surge_pwm = self.pid_to_pwm(surge_control_output)

                    # Compute error derivative for potential stuck detection
                    # This logic could be simplified or improved
                    self.pinger_error = pinger_distance - self.pinger_threshold
                    # Calculate time delta since last PID update or loop iteration
                    time_delta = self.get_clock().now().seconds_nanoseconds()[0] + \
                                 self.get_clock().now().seconds_nanoseconds()[1] * 1e-9 - current_time # This is not reliable dt
                    # A better way is to store last time and calculate dt

                    # If the distance isn't changing much AND we are too close, assume stuck and search
                    # This logic needs refinement, error_change alone might not be enough
                    # if abs(self.pinger_error_change) < 0.005 and pinger_distance < self.pinger_threshold:
                    #     self.current_pinger_state = "stuck_searching"
                    #     self.surge_pwm = 1500
                    #     self.sway_pwm = 1500
                    #     self.search_path = True


                # If currently searching for a path
                if self.search_path:
                    self.current_pinger_state = "searching"
                    self.surge_pwm = 1500 # Stop surge while searching
                    # Lateral movement for searching (crab walk or rotate)
                    if self.crab_walk: # Assuming crab_walk is set externally or decided here
                        self.sway_pwm = 1600 # Example: Crab walk right
                        self.Correction_yaw_pwm = 1500 # Keep yaw neutral
                        self.get_logger().info("Searching: Crabbing Right")
                        # Add logic to stop crabbing and check again after a duration
                    else: # Rotate to search
                        # Rotate slowly to look for a clear path
                        # Update desired_yaw for the yaw PID (running in OdoCallback)
                        rotation_speed_degrees_per_sec = 5.0 # Example speed
                        yaw_delta = rotation_speed_degrees_per_sec * self.control_period # Calculate rotation step

                        if self.positive_rotation:
                            self.desired_yaw += yaw_delta
                            if self.desired_yaw > 180: # Wrap around
                                self.desired_yaw -= 360
                        else:
                            self.desired_yaw -= yaw_delta
                            if self.desired_yaw < -180: # Wrap around
                                self.desired_yaw += 360

                        # The yaw PID in OdoCallback will try to reach this desired_yaw
                        self.sway_pwm = 1500 # Keep sway neutral while rotating
                        self.get_logger().info(f"Searching: Rotating. Desired Yaw: {self.desired_yaw:.2f}")


                    # If obstacle is now far enough, assume free path
                    if pinger_distance > 2.0 * self.pinger_threshold: # Use a hysteresis threshold
                         self.free_path = True
                         self.search_path = False # Exit search mode
                         self.current_pinger_state = "free_path_detected"
                         self.get_logger().info("Free path detected by pinger.")
                         # Continue to free_path logic in the next iteration

                # If the path is clear, move forward
                if self.free_path and not self.search_path:
                    self.current_pinger_state = "moving_forward"
                    # Move forward at a cautious speed
                    self.surge_pwm = 1535 # Example speed (adjust as needed)
                    self.sway_pwm = 1500   # Keep lateral movement neutral
                    # Yaw should ideally hold the last known good heading or face the target if known
                    # For simplicity, let the yaw PID maintain the current heading or a target set elsewhere
                    # self.Correction_yaw_pwm is controlled by OdoCallback's PID on self.desired_yaw


                    # Add a condition to exit the action if the mission goal is reached
                    # For example, if distance > some very large value or external signal received
                    # For now, let's say it succeeds after moving forward for a bit or if distance stays high
                    # This needs a proper mission control signal. As a placeholder, let's say it succeeds
                    # if distance remains high for a certain duration.
                    # This part is crucial: When does the ACTION *finish* successfully?
                    # A simple example: If distance > threshold and state is moving_forward
                    if pinger_distance > self.pinger_threshold * 1.5: # Still good distance
                         # Consider the goal achieved if we successfully moved forward past the obstacle
                         # This condition needs to be more robust, e.g., based on mission waypoints or time
                         self.get_logger().info("Pinger avoidance successful.")
                         result.success = True
                         result.message = "Obstacle avoided, path clear"
                         # Reset PWMs to neutral on success
                         self.surge_pwm = 1500
                         self.sway_pwm = 1500
                         self.Correction_yaw_pwm = 1500 # Also reset yaw in this mode
                         goal_handle.succeed()
                         return result


            # --- End Pinger Avoidance/Search Logic ---


            # Publish feedback
            feedback_msg.current_state = self.current_pinger_state
            feedback_msg.current_pinger_distance = pinger_distance
            feedback_msg.current_pinger_confidence = pinger_confidence
            goal_handle.publish_feedback(feedback_msg)

            # Sleep to control loop rate
            rate.sleep()

        # If the loop exits without success or cancel (e.g., rclpy.ok() becomes false)
        result.success = False
        result.message = "Pinger avoidance aborted unexpectedly"
        # Reset PWMs
        self.surge_pwm = 1500
        self.sway_pwm = 1500
        self.Correction_yaw_pwm = 1500
        goal_handle.abort()
        return result


    # Helper to get current yaw angle in degrees (assuming OdoCallback updates angle.angular.z)
    def get_yaw_degrees(self):
         # Access the latest calculated yaw from OdoCallback if stored, or recalculate if necessary
         # Assuming angle_wrt_startup[2] is updated by OdoCallback and stored implicitly
         # A safer way is to store the latest angle message received in OdoCallback
         # For now, let's assume self.angle_yaw_a0 and the calculation is accessible/updated
         # This is error-prone if OdoCallback doesn't run frequently or consistently update state for this method
         # Best practice: Store the latest angle message data in a member variable in OdoCallback
         # And access that member variable here.
         # Let's assume for now OdoCallback's data is accessible or can be re-derived roughly

         # Placeholder: You need to store the latest angle message in OdoCallback
         # self.latest_imu_angle = angle # in OdoCallback
         # current_yaw = self.latest_imu_angle.angular.z if hasattr(self, 'latest_imu_angle') else 0.0

         # Re-using the OdoCallback's logic for simplicity, but less robust
         # Need latest orientation data from IMU subscription
         # self.latest_orientation = data.orientation in OdoCallback
         if hasattr(self, 'latest_orientation'):
             x = self.latest_orientation.x
             y = self.latest_orientation.y
             z = self.latest_orientation.z
             w = self.latest_orientation.w
             siny_cosp = 2.0 * (w * z + x * y)
             cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
             angle_yaw = np.arctan2(siny_cosp, cosy_cosp)
             # Convert relative to startup yaw
             angle_wrt_startup_yaw = ((angle_yaw - self.angle_yaw_a0 + 3.0 * math.pi) % (
                         2.0 * math.pi) - math.pi) * 180 / math.pi
             return angle_wrt_startup_yaw
         else:
             return 0.0 # Return 0 or raise error if IMU data hasn't arrived

    # --- End Action Server Callbacks ---


    def trajectory_callback(self, request, response):
        """
        Service callback: Start or stop trajectory generation based on boolean input.
        (Depth Trajectory - separate from Pinger Action)
        """
        if request.data:  # True -> Start trajectory
            # Cancel Pinger Action if active when starting trajectory
            if self._pinger_avoidance_goal_handle and self._pinger_avoidance_goal_handle.is_active:
                self.get_logger().info("Cancelling PingerAvoidance action before starting trajectory.")
                cancel_request = rclpy.action.CancelGoal.Request()
                # Assuming this node is also the client requesting the action
                # A proper client node would send the cancel request
                # For this structure, the server needs to handle its own internal state change
                # Setting goal_handle to canceled internally is sufficient here
                if self._pinger_avoidance_goal_handle:
                     self._pinger_avoidance_goal_handle.canceled()
                     self._pinger_avoidance_goal_handle = None # Clear the handle
                     self.get_logger().info("PingerAvoidance action internally cancelled.")
                     # Reset PWMs managed by action
                     self.surge_pwm = 1500
                     self.sway_pwm = 1500
                     self.Correction_yaw_pwm = 1500


            self.time_init = self.get_clock().now().seconds_nanoseconds()[0] + \
                             self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
            # Make sure CubicTrajectory is initialized or available here
            # For now, assume it's available but commented out
            # self.trajectory = CubicTrajectory(z_init=self.depth_p0, z_final=-0.2)
            self.time_final = self.time_init + 20  # 20 seconds trajectory
            self.traj_active = True  # Enable trajectory following
            response.success = True
            response.message = "Trajectory started"
            self.get_logger().info("Trajectory started.")
        else:  # False -> Stop trajectory
            self.traj_active = False
            response.success = True
            response.message = "Trajectory stopped"
            self.get_logger().info("Trajectory stopped.")

        return response

    def pid_to_pwm(self, pid):
        """
        Convert pid output to pwm signal
        """
        # Ensure pid is a float or int before comparison
        pid = float(pid)
        if pid > 0:
            pwm = 1500 - 110 * pid
        else:
            pwm = 1500 - 90 * pid
        # Ensure conversion to integer for pulse width
        pwm = int(pwm)
        if pwm > 1900:
            pwm = 1900
        elif pwm < 1100:
            pwm = 1100
        return float(pwm) # Return float as OverrideRCIn expects uint, conversion happens later

    def RelAltCallback(self, data):
        """
        Get depth sensor data from this function
        """
        if (self.init_p0):
            # 1st execution, init
            self.depth_p0 = data.data
            self.init_p0 = False

        time_tupple = self.get_clock().now().seconds_nanoseconds()
        current_time = time_tupple[0] + (time_tupple[1] * 10**-9)

        current_depth = data.data

        # check if depth trajectory is generated/active
        if self.traj_active:
            # Get waypoint from trajectory
            # Make sure self.trajectory object is initialized and has get_waypoint method
            # For now, assuming commented out, so manual desired_depth or service controls it
            # If uncommented, use this logic:
            self.desired_depth, desired_velocity = self.trajectory.get_waypoint(current_time, self.time_init, self.time_final)

            # Publish waypoint (if trajectory is active)
            waypoint_msg = Pose()
            waypoint_msg.position.z = self.desired_depth
            self.pub_generated_traj.publish(waypoint_msg)

            waypoint_dot_msg = Twist()
            waypoint_dot_msg.linear.z = desired_velocity
            self.pub_generated_traj_dot.publish(waypoint_dot_msg)

            self.get_logger().info(f"Generated Waypoint - Z: {self.desired_depth:.3f}, Z_dot: {desired_velocity:.3f}")

            # Stop trajectory if time exceeds
            if current_time > self.time_final:
                self.traj_active = False
                self.get_logger().info("Trajectory complete.")

        ##########################################
        # depth_control = 0.37 # floatability of the robot
        ##########################################
        floatability = 0.305 # This is likely used in the PID output adjustment

        # Only run depth PID if not in manual mode (mode 0)
        if not self.set_mode[0]:
             depth_control_output = self.pid_depth.calculate_pid(self.desired_depth, current_depth, current_time)
             # Adjust PID output based on floatability
             depth_control_output -= floatability # Or adjust based on the PID type and how floatability affects error

             # Convert PID output to PWM
             depth_pwm_val = self.pid_to_pwm(-depth_control_output) # PID output needs sign adjustment?

             # Store the calculated depth PWM for timer_callback
             self.Correction_depth = int(depth_pwm_val)

             pub_error_depth = Float64()
             pub_error_depth.data = depth_control_output # Publish the PID output before PWM conversion
             self.pub_depth_error.publish(pub_error_depth)

             pub_depth = Float64()
             pub_depth.data = float(self.Correction_depth) # Publish the final PWM value
             self.pub_depth_pwm.publish(pub_depth)
        else:
             # In manual mode, depth PWM might be controlled by joystick directly
             # Or keep it neutral if not receiving joystick commands
             self.Correction_depth = 1500 # Default to neutral in manual mode


    def OdoCallback(self, data):
        """
        Get imu data from this function
        """
        # Store latest orientation data for get_yaw_degrees
        self.latest_orientation = data.orientation

        # get time now
        time_tupple = self.get_clock().now().seconds_nanoseconds()
        current_time = time_tupple[0] + (time_tupple[1] * 10**-9)

        orientation = data.orientation
        angular_velocity = data.angular_velocity

        # extraction of roll, pitch, yaw angles
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w

        # quaternion to euler angles
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        sinp = 2.0 * (w * y - z * x)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        angle_roll = np.arctan2(sinr_cosp, cosr_cosp)
        angle_pitch = np.arcsin(sinp)
        angle_yaw = np.arctan2(siny_cosp, cosy_cosp)

        if (self.init_a0):
            # at 1st execution, init
            self.angle_roll_a0 = angle_roll
            self.angle_pitch_a0 = angle_pitch
            self.angle_yaw_a0 = angle_yaw
            self.init_a0 = False

        # Calculate angles relative to startup
        angle_wrt_startup = [0] * 3
        angle_wrt_startup[0] = ((angle_roll - self.angle_roll_a0 + 3.0 * math.pi) % (
                    2.0 * math.pi) - math.pi) * 180 / math.pi
        angle_wrt_startup[1] = ((angle_pitch - self.angle_pitch_a0 + 3.0 * math.pi) % (
                    2.0 * math.pi) - math.pi) * 180 / math.pi
        angle_wrt_startup[2] = ((angle_yaw - self.angle_yaw_a0 + 3.0 * math.pi) % (
                    2.0 * math.pi) - math.pi) * 180 / math.pi

        angle = Twist() # orientation in degrees but using twist msg for some reason
        angle.angular.x = angle_wrt_startup[0]
        angle.angular.y = angle_wrt_startup[1]
        angle.angular.z = angle_wrt_startup[2] # This is the current yaw in degrees relative to startup

        self.pub_angle_degree.publish(angle)

        # Extraction of angular velocity
        p = angular_velocity.x
        q = angular_velocity.y
        r = angular_velocity.z
        vel = Twist() # angular velocity
        vel.angular.x = p
        vel.angular.y = q
        vel.angular.z = r

        # publish angular velocity
        self.pub_angular_velocity.publish(vel)

        # --- Yaw Control (Active in modes 2 and 3) ---
        # The desired_yaw might be set by the action execution (mode 3)
        # or set to a default value (mode 2) or manual input (mode 0)

        # Only run yaw PID if in corrected mode (2) or pinger mode (3)
        if self.set_mode[2] or self.set_mode[3]:

            # Compute yaw error using the current yaw and desired_yaw
            # Note: Ensure desired_yaw is also in degrees relative to startup or handle wrapping
            yaw_error_degrees = self.desired_yaw - angle.angular.z
            # Wrap error to [-180, 180] degrees
            if yaw_error_degrees > 180:
                 yaw_error_degrees -= 360
            elif yaw_error_degrees < -180:
                 yaw_error_degrees += 360

            # yaw_control is the PID output
            yaw_control_output = self.pid_yaw.calculate_pid(self.desired_yaw, angle.angular.z, current_time)

            # Convert PID output to PWM
            yaw_pwm_val = self.pid_to_pwm(yaw_control_output) # Adjust sign if needed

            # Store the calculated yaw PWM for timer_callback
            self.Correction_yaw_pwm = int(yaw_pwm_val)

            pub_error_yaw = Float64()
            pub_error_yaw.data = yaw_control_output # Publish PID output
            self.yaw_error_pwm.publish(pub_error_yaw)

        elif self.set_mode[0] or self.set_mode[1]:
            # In manual or auto (cmd_vel) mode, yaw is controlled by joystick/cmd_vel
            # Keep Correction_yaw_pwm neutral or let velCallback/joyCallback set it directly
            pass # joyCallback/velCallback will set it via setOverrideRCIN

    # The pinger_callback now only receives and stores the latest data
    def pinger_callback(self, data):
        # Only store data if we are potentially in a mode that uses it (mode 3)
        # The action execution will read these stored values
        if self.set_mode[3]:
             self.latest_pinger_distance = data.data[0]
             self.latest_pinger_confidence = data.data[1]
             # self.get_logger().info(f"Pinger Data - Distance: {self.latest_pinger_distance:.3f}, Confidence: {self.latest_pinger_confidence}")
        # else:
        #      # Optional: Log if data is received but not used
        #      self.get_logger().debug("Pinger data received but not in pinger mode.")


    def crab_walk_callback(self, data):
        # This flag is used by the pinger avoidance action logic
        self.crab_walk = data.data

    def timer_callback(self):
        # This timer callback is responsible for sending the latest calculated PWM commands
        # The values in self.Correction_depth, self.Correction_yaw_pwm, self.surge_pwm, self.sway_pwm
        # are updated by the respective controllers (PID for depth/yaw, Action for surge/sway/yaw in mode 3)
        # or by the joystick/cmd_vel callbacks (modes 0/1).

        # Only send override commands if we are not in manual mode being controlled by joystick/cmd_vel directly
        # Modes 2 and 3 rely on this timer to send PID/Action calculated values.
        # Mode 1 (cmd_vel) could also use this if velCallback updates class members
        # Mode 0 (manual joystick) uses setOverrideRCIN directly in joyCallback.

        # Check if in a mode where the timer should send commands based on class members
        if self.set_mode[2] or self.set_mode[3] or self.set_mode[1]: # Added mode 1 if it updates members
             self.setOverrideRCIN(1500, 1500, # Pitch, Roll (assuming not controlled by PID/Action here)
                                 self.Correction_depth,
                                 self.Correction_yaw_pwm,
                                 self.surge_pwm,
                                 self.sway_pwm)
             # self.get_logger().info(f"Timer publishing - Depth: {self.Correction_depth}, Yaw: {self.Correction_yaw_pwm}, Surge: {self.surge_pwm}, Sway: {self.sway_pwm}")

        # else: # Mode 0 (Manual)
        #     # setOverrideRCIN is called directly by joyCallback in manual mode
        #     pass


    def armDisarm(self, armed):
        # This functions sends a long command service with 400 code to arm or disarm motors
        # ... (rest of the armDisarm function remains the same)
        self.get_logger().info(f"Arm/Disarm service called with armed={armed}") # Added log
        traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
        cli = self.create_client(CommandLong, 'cmd/command')
        result = False
        while not result:
            result = cli.wait_for_service(timeout_sec=4.0)
            self.get_logger().info(f"{'arming' if armed else 'disarming'} requested, wait_for_service, timeout, result : {result}")
        req = CommandLong.Request()
        req.broadcast = False
        req.command = 400
        req.confirmation = 0
        req.param1 = 1.0 if armed else 0.0
        req.param2 = 0.0
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = 0.0
        req.param6 = 0.0
        req.param7 = 0.0
        self.get_logger().info("just before call_async")
        resp = cli.call_async(req)
        # Note: spin_until_future_complete blocks, might be better to use futures
        # rclpy.spin_until_future_complete(self, resp)
        self.get_logger().info(f"{'Arming' if armed else 'Disarming'} request sent.")
        # You might want to add code here to handle the response once it completes.


    def manageStabilize(self, stabilized):
        # This functions sends a SetMode command service to stabilize or reset
        # ... (rest of the manageStabilize function remains the same)
         self.get_logger().info(f"SetMode service called with stabilized={stabilized}") # Added log
         traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
         cli = self.create_client(SetMode, 'set_mode')
         result = False
         while not result:
             result = cli.wait_for_service(timeout_sec=4.0)
             self.get_logger().info(f"{'stabilized' if stabilized else 'manual'} mode requested, wait_for_service, (False if timeout) result : {result}")

         req = SetMode.Request()
         req.base_mode = 0
         req.custom_mode = "0" if stabilized else "19" # "0" for STABILIZE, "19" for MANUAL
         resp = cli.call_async(req)
         # rclpy.spin_until_future_complete(self, resp)
         self.get_logger().info(f"Set mode to {'STABILIZE' if stabilized else 'MANUAL'} request sent.")
         # Handle response if needed


    def setStreamRate(self, rate):
        # ... (rest of the setStreamRate function remains the same)
        self.get_logger().info(f"SetStreamRate service called with rate={rate}") # Added log
        traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
        cli = self.create_client(StreamRate, 'set_stream_rate')
        result = False
        while not result:
            result = cli.wait_for_service(timeout_sec=4.0)
            self.get_logger().info("stream rate requested, wait_for_service, (False if timeout) result :" + str(result))

        req = StreamRate.Request()
        req.stream_id = 0 # Stream_id 0 is ALL_DATA
        req.message_rate = rate
        req.on_off = True
        resp = cli.call_async(req)
        rclpy.spin_until_future_complete(self, resp) # This blocks
        self.get_logger().info("set stream rate Succeeded")


    def addEndPoint(self):
        # ... (rest of the addEndPoint function remains the same)
        self.get_logger().info("AddEndPoint service called") # Added log
        traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
        cli = self.create_client(EndpointAdd, 'mavros_router/add_endpoint')
        result = False
        while not result:
            result = cli.wait_for_service(timeout_sec=4.0)
            self.get_logger().info(
                "add endpoint requested, wait_for_service, (False if timeout) result :" + str(result))

        req = EndpointAdd.Request()
        req.url = "udp://@localhost:14550" # Default MAVROS GCS port
        req.type = 1  # TYPE_GCS
        resp = cli.call_async(req)
        rclpy.spin_until_future_complete(self, resp) # This blocks
        self.get_logger().info("add endpoint rate Succeeded")


    def joyCallback(self, data):
        # Joystick buttons
        btn_arm = data.buttons[7]  # Start button
        btn_disarm = data.buttons[6]  # Back button
        btn_manual_mode = data.buttons[3]  # Y button
        btn_automatic_mode = data.buttons[2]  # X button (cmd_vel)
        btn_corrected_mode = data.buttons[0]  # A button (Depth/Yaw PID)
        btn_collision_mode = data.buttons[1]  # B button (Pinger Action)

        # Disarming when Back button is pressed
        if (btn_disarm == 1 and self.arming == True):
            self.arming = False
            self.armDisarm(self.arming)
            # Cancel any active action/mode when disarming
            self.cancel_active_actions_and_reset_modes()

        # Arming when Start button is pressed
        if (btn_arm == 1 and self.arming == False):
            self.arming = True
            self.armDisarm(self.arming)
            # Maybe set a default mode upon arming, e.g., manual
            self.set_mode = [True, False, False, False]
            self.get_logger().info("Set mode to Manual on arming")
            self.cancel_active_actions_and_reset_modes() # Ensure cleanup

        # --- Mode Switching ---
        # Cancel any active action/mode before switching
        prev_mode = self.set_mode[:] # Copy current modes
        new_mode = prev_mode[:]

        mode_changed = False

        if (btn_manual_mode == 1 and not self.set_mode[0]):
            new_mode = [True, False, False, False]
            self.get_logger().info("Switching to Mode manual")
            mode_changed = True
        elif (btn_automatic_mode == 1 and not self.set_mode[1]):
            new_mode = [False, True, False, False]
            self.get_logger().info("Switching to Mode automatic (cmd_vel)")
            mode_changed = True
        elif (btn_corrected_mode == 1 and not self.set_mode[2]):
            self.init_a0 = True # Re-initialize angle reference on entering corrected mode
            self.init_p0 = True # Re-initialize depth reference
            # set sum errors to 0 here, ex: Sum_Errors_Vel = [0]*3
            self.pid_depth.reset_integral() # Reset PID integrals on mode change
            self.pid_yaw.reset_integral()
            new_mode = [False, False, True, False]
            self.get_logger().info("Switching to Mode correction (PID depth/yaw)")
            mode_changed = True
        elif (btn_collision_mode == 1 and not self.set_mode[3]):
             # Re-initialize angle/depth references is less critical here as PIDs use relative values
             # But resetting PID integrals is good practice
            #  self.pid_surge.reset_integral() # Reset surge PID integral
            #  self.pid_sway.reset_integral() # Reset sway PID integral
             # Yaw PID integral might or might not need reset depending on how desired_yaw is set
            #  self.pid_yaw.reset_integral()
             new_mode = [False, False, False, True]
             self.get_logger().info("Switching to Mode collision avoidance (Pinger Action)")
             mode_changed = True

        if mode_changed:
            self.set_mode = new_mode
            self.cancel_active_actions_and_reset_modes() # Ensure cleanup before starting new mode logic

            # If switching TO Pinger mode, start the action
            if self.set_mode[3]:
                self.start_pinger_avoidance_action()



    def cancel_active_actions_and_reset_modes(self):
        """Cancels any active actions and resets relevant PWM values."""
        self.get_logger().info("Cancelling active actions and resetting modes.")
        # Cancel the PingerAvoidance action if it's active
        if self._pinger_avoidance_goal_handle and self._pinger_avoidance_goal_handle.is_active:
            self._pinger_avoidance_goal_handle.canceled() # Inform the action it's cancelled
            # The execute_callback should handle the cancellation request and cleanup PWMs
            self._pinger_avoidance_goal_handle = None # Clear the handle

        # Cancel the Depth Trajectory if active
        if self.traj_active:
             self.traj_active = False
             self.get_logger().info("Depth Trajectory stopped.")
             # Depth PWM is handled by PID, which will go to neutral (1500) if desired=current

        # Reset all control-related PWMs to neutral (1500) as a fallback
        # The timer_callback or specific mode logic will set these again if needed
        self.Correction_depth = 1500
        self.Correction_yaw_pwm = 1500
        self.surge_pwm = 1500
        self.sway_pwm = 1500

        # Reset PID integrals on mode change
        self.pid_depth.reset_integral()
        self.pid_yaw.reset_integral()
        self.pid_surge.reset_integral()
        self.pid_sway.reset_integral()

        # Reset desired states
        self.desired_depth = self.depth_p0 # Reset desired depth to initial depth
        self.desired_yaw = self.get_yaw_degrees() # Reset desired yaw to current yaw

        self.current_pinger_state = "idle" # Reset pinger state feedback


    def start_pinger_avoidance_action(self):
        """Starts the PingerAvoidance action."""
        # Check if an action is already running
        if self._pinger_avoidance_goal_handle and self._pinger_avoidance_goal_handle.is_active:
            self.get_logger().info("PingerAvoidance action is already active.")
            return

        # Create a new goal
        goal_msg = PingerAvoidance.Goal()
        goal_msg.start_avoidance = True # Example goal parameter

        self.get_logger().info('Sending PingerAvoidance goal request...')

        # Action clients are typically separate, but for triggering from inside the node
        # we can conceptualize this as an internal trigger to the server.
        # A proper client would create a client object and send the goal.
        # Since this node IS the server, we just need to set the mode and the
        # timer_callback will rely on the action's execute_callback to update PWMs.
        # The execute_callback runs in a separate thread when a goal is accepted.

        # We already set self.set_mode[3] = True in joyCallback.
        # The execute_callback will start running because handle_goal accepted the request.
        # The timer_callback will then use the PWMs calculated in execute_callback.

        # Note: In a more complex system, you might have a dedicated PingerClient node
        # that sends the goal to this node's ActionServer. But for integrating the
        # logic into the existing node, this approach of triggering via mode switch
        # and relying on the server's execute_callback to update state is common.
        pass # The action server is already running and will process the request


    def velCallback(self, cmd_vel):
        # Only continue if automatic mode (cmd_vel) is enabled (mode 1)
        if not self.set_mode[0]:
            return
        else:
            self.get_logger().info("Receiving cmd_vel and sending manual-like commands.")

        # Extract cmd_vel message and convert to PWM using scaling/saturation
        # This assumes cmd_vel provides values in a range like -1.0 to 1.0 or similar
        # Adjust scaling if your cmd_vel messages use different units (e.g., m/s, rad/s)

        # Need to map cmd_vel fields to PWM values (1100-1900)
        # Assuming cmd_vel linear/angular fields are scaled (e.g., -1.0 to 1.0 for full power)
        pwm_roll = self.mapValueScalSat(cmd_vel.angular.x)
        pwm_yaw = self.mapValueScalSat(-cmd_vel.angular.z) # Assuming negative for left yaw
        pwm_ascend_descend = self.mapValueScalSat(cmd_vel.linear.z)
        pwm_forward_reverse = self.mapValueScalSat(cmd_vel.linear.x)
        pwm_lateral_left_right = self.mapValueScalSat(-cmd_vel.linear.y) # Assuming negative for left sway
        pwm_pitch = self.mapValueScalSat(cmd_vel.angular.y)


        # In this mode (set_mode[1]), we update the class members based on cmd_vel
        # The timer_callback will then read these and send the OverrideRCIn message.
        # This decoupling allows the timer to run at a fixed rate regardless of cmd_vel arrival rate.
        self.Correction_pitch_pwm = pwm_pitch # Assuming you add this member or use channel 0 in setOverrideRCIN
        self.Correction_roll_pwm = pwm_roll   # Assuming you add this member or use channel 1
        self.Correction_depth = pwm_ascend_descend # Z linear
        self.Correction_yaw_pwm = pwm_yaw # Z angular
        self.surge_pwm = pwm_forward_reverse # X linear
        self.sway_pwm = pwm_lateral_left_right # Y linear
        self.setOverrideRCIN(pwm_pitch, pwm_roll, pwm_ascend_descend, pwm_yaw, pwm_forward_reverse,
                                 pwm_lateral_left_right)
        # Ensure Correction_pitch_pwm and Correction_roll_pwm are initialized in __init__
        # and used in setOverrideRCIN call in timer_callback


    def visual_tracker_callback(self, data):
        # Only continue if visual tracker mode is active (you need to define a mode for this)
        # Let's assume a new mode, say mode 4, or integrate into mode 2/3 if appropriate.
        # If integrating into mode 2/3, add checks like `if self.set_mode[2] and visual_tracker_enabled:`
        # Or, if this callback *itself* should directly send commands, it needs its own mode.
        # Based on your original code, it seems to overlap with modes 0 and 2?
        # Let's assume it should ONLY work if the mode is specifically for visual tracking,
        # distinct from manual (0), cmd_vel (1), corrected (2), or pinger (3).
        # Or maybe visual tracking *augments* one of the modes?

        # For now, let's assume it operates in a dedicated mode, or is integrated conditionally.
        # If it's a separate mode, update joyCallback to select it.
        # If it augments, e.g., corrected mode (2), add `if self.set_mode[2]:` check.

        # As per the original code structure, it seems to bypass modes 0 and 2.
        # This suggests it might be intended for mode 1 or 3, or a dedicated mode.
        # Let's assume it's a separate controller logic that could apply in modes 1 or 2 or 3
        # but only when visual tracking is active and the mode allows it.
        # This requires more complex state management.

        # Let's stick to the original check for now, but note its ambiguity
        if (self.set_mode[2] or self.set_mode[0] or self.set_mode[3]): # Original check - might be wrong based on intent
            self.get_logger().debug("Not in correct mode for visual tracker, ignoring data.")
            return
        else:
            self.get_logger().info("Visual tracker data received, processing.")
            # Map visual tracker output (assuming it's scaled like cmd_vel) to PWM
            pwm_roll = self.mapValueScalSat(data.angular.x)
            pwm_yaw = self.mapValueScalSat(data.angular.z) # Assuming sign is correct
            pwm_ascend_descend = self.mapValueScalSat(data.linear.z)
            pwm_forward_reverse = self.mapValueScalSat(data.linear.x)
            pwm_lateral_left_right = self.mapValueScalSat(data.linear.y) # Assuming sign is correct
            pwm_pitch = self.mapValueScalSat(data.angular.y)

            # Apply saturation limits as in the original code
            pwm_yaw = min(1600, max(1400, pwm_yaw))  # Saturate yaw command
            pwm_lateral_left_right = min(1600, max(1400, pwm_lateral_left_right))  # Saturate lateral command

            # Directly send the commands if this callback is the control source for the current mode
            # This bypasses the timer_callback and other controllers for these specific channels.
            # This suggests this callback *is* the controller for its intended mode.
            # Need to be careful not to conflict with timer_callback sending data for other modes.

            # Let's revise: If this callback *should* be the controller, it needs its own mode check.
            # Assuming it's intended for a mode where visual tracking provides the commands:
            # Add a dedicated mode, e.g., mode 4 for visual tracking.
            # Then in joyCallback, add `elif (btn_visual_track == 1 and not self.set_mode[4]): new_mode = [False, False, False, False, True]` etc.
            # And here: `if self.set_mode[4]: self.setOverrideRCIN(...)`

            # Given the original structure, it seems this callback might have been intended to
            # augment or *be* the control in modes other than 0 and 2. Let's update the check.
            # Assuming it's for mode 1 (cmd_vel replacement) or mode 3 (pinger replacement/augmentation)
            # This needs clarification based on intended behavior.

            # Let's assume this callback should update the same PWM members that the timer sends,
            # just like velCallback does for mode 1. This allows timer_callback to consistently send.
            if self.set_mode[1]: # Apply visual tracking in mode 1 or 3? (Example)
                self.get_logger().info("Applying visual tracker commands to PWM members.")
                # Update class members that timer_callback reads
                self.Correction_pitch_pwm = pwm_pitch
                self.Correction_roll_pwm = pwm_roll
                self.Correction_depth = pwm_ascend_descend
                self.Correction_yaw_pwm = pwm_yaw # Overwrites yaw PID if active in mode 3
                self.surge_pwm = pwm_forward_reverse # Overwrites surge PID if active in mode 3
                self.sway_pwm = pwm_lateral_left_right # Overwrites sway PID if active in mode 3
            # else:
                 # self.get_logger().debug("Visual tracker data received, but not in a mode that uses it.")


    def setOverrideRCIN(self, channel_pitch, channel_roll, channel_throttle, channel_yaw, channel_forward,
                        channel_lateral):
        # This function replaces setservo for motor commands.
        # It overrides Rc channels inputs and simulates motor controls.
        # In this case, each channel manages a group of motors not individually as servo set

        msg_override = OverrideRCIn()
        # Ensure all values are integers before packing
        msg_override.channels[0] = np.uint(int(channel_pitch))  # pitch
        msg_override.channels[1] = np.uint(int(channel_roll))  # roll
        msg_override.channels[2] = np.uint(int(channel_throttle))  # heave (depth)
        msg_override.channels[3] = np.uint(int(channel_yaw))  # yaw
        msg_override.channels[4] = np.uint(int(channel_forward))  # surge
        msg_override.channels[5] = np.uint(int(channel_lateral))  # sway
        msg_override.channels[6] = 1500 # Aux 1
        msg_override.channels[7] = 1500 # Aux 2

        self.pub_msg_override.publish(msg_override)
        # self.get_logger().info(f"Sending OverrideRCIn: Thr={channel_throttle}, Yaw={channel_yaw}, Surge={channel_forward}, Sway={channel_lateral}") # Detailed logging


    def mapValueScalSat(self, value):
        # Correction_Vel and joy between -1 et 1
        # scaling for publishing with setOverrideRCIN values between 1100 and 1900
        # neutral point is 1500
        pulse_width = float(value) * 400.0 + 1500.0 # Ensure float multiplication

        # Saturation
        if pulse_width > 1900:
            pulse_width = 1900
        if pulse_width < 1100:
            pulse_width = 1100

        return int(pulse_width) # Return integer for PWM


    def subscriber(self):
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subjoy = self.create_subscription(Joy, "joy", self.joyCallback, qos_profile=qos_profile)
        # self.subjoy  # prevent unused variable warning (Pylint/Flake8)
        self.subcmdvel = self.create_subscription(Twist, "cmd_vel", self.velCallback, qos_profile=qos_profile)
        # self.subcmdvel  # prevent unused variable warning
        self.subimu = self.create_subscription(Imu, "imu/data", self.OdoCallback, qos_profile=qos_profile)
        # self.subimu  # prevent unused variable warning

        # visual tracker callback
        # Note: Check the actual topic name
        self.subvisual_tracker = self.create_subscription(Twist, "visual_tracker", self.visual_tracker_callback,
                                                           qos_profile=qos_profile)
        # self.subvisual_tracker  # prevent unused variable warning

        # Note: Check the actual topic name for relative altitude/depth
        self.subrel_alt = self.create_subscription(Float64, "global_position/rel_alt", self.RelAltCallback,
                                                   qos_profile=qos_profile)
        # self.subrel_alt  # prevent unused variable warning

        # Note: Check the actual topic name for pinger data
        self.sublaser = self.create_subscription(Float64MultiArray, '/bluerov2/ping1d/data', self.pinger_callback,
                                                 qos_profile=qos_profile)
        # self.sublaser

        # crab walk flag subscription
        self.subcrab_walk = self.create_subscription(Bool, '/crab_walk', self.crab_walk_callback,
                                                     qos_profile=qos_profile)
        # self.subcrab_walk


        self.get_logger().info("Subscriptions done.")


    ### For PID Controller Parameters ###
    def declare_and_set_params(self):
        """Declares PID parameters with descriptions and ranges."""
        # Depth PID Parameters
        self._declare_and_fill_slider('k_p_depth', 2.0, "K P of depth", 0.0, 10.0, self.config) # last kp val 4
        self._declare_and_fill_slider('k_i_depth', 0.0, "K I of depth", 0.0, 5.0, self.config)
        self._declare_and_fill_slider('k_d_depth', 0.0, "K D of depth", 0.0, 5.0, self.config)

        self._declare_and_fill_slider('k_p_yaw', 0.1, "K P of yaw", 0.0, 10.0, self.config)
        self._declare_and_fill_slider('k_i_yaw', 0.0, "K I of yaw", 0.0, 5.0, self.config)
        self._declare_and_fill_slider('k_d_yaw', 0.0, "K D of yaw", 0.0, 5.0, self.config)

        self._declare_and_fill_slider('k_p_surge', 0.0, "K P of surge", 0.0, 10.0, self.config)
        self._declare_and_fill_slider('k_i_surge', 0.0, "K I of surge", 0.0, 5.0, self.config)
        self._declare_and_fill_slider('k_d_surge', 0.0, "K D of surge", 0.0, 5.0, self.config)

        self._declare_and_fill_slider('k_p_sway', 0.0, "K P of sway", 0.0, 10.0, self.config)
        self._declare_and_fill_slider('k_i_sway', 0.0, "K I of sway", 0.0, 5.0, self.config)
        self._declare_and_fill_slider('k_d_sway', 0.0, "K D of sway", 0.0, 5.0, self.config)

        # Pinger Threshold Parameter
        self.declare_parameter('pinger_threshold', 0.75, ParameterDescriptor(description='Pinger distance threshold for avoidance'))


        # Get initial parameters
        self.update_control_param()

    def _declare_and_fill_slider(self, key, default_value, description, min_value, max_value, map):
        float_range = FloatingPointRange(from_value=min_value, to_value=max_value, step=0.0)
        param = self.declare_parameter(
            key, default_value, ParameterDescriptor(
                description=description,
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[float_range]
            )
        )
        map[key] = param.value
        
    def callback_params(self, params):
        """Callback for parameter changes."""
        for param in params:
            if param.name == 'k_p_depth':
                self.config['k_p_depth'] = param.value
                self.get_logger().info(f"Updated k_p_depth to: {param.value}")
            elif param.name == 'k_i_depth':
                self.config['k_i_depth'] = param.value
                self.get_logger().info(f"Updated k_i_depth to: {param.value}")
            elif param.name == 'k_d_depth':
                self.config['k_d_depth'] = param.value
                self.get_logger().info(f"Updated k_d_depth to: {param.value}")
            elif param.name == 'k_p_yaw':
                self.config['k_p_yaw'] = param.value
                self.get_logger().info(f"Updated k_p_yaw to: {param.value}")
            elif param.name == 'k_i_yaw':
                self.config['k_i_yaw'] = param.value
                self.get_logger().info(f"Updated k_i_yaw to: {param.value}")
            elif param.name == 'k_d_yaw':
                self.config['k_d_yaw'] = param.value
                self.get_logger().info(f"Updated k_d_yaw to: {param.value}")
            elif param.name == 'k_p_surge':
                self.config['k_p_surge'] = param.value
                self.get_logger().info(f"Updated k_p_surge to: {param.value}")
            elif param.name == 'k_i_surge':
                self.config['k_i_surge'] = param.value
                self.get_logger().info(f"Updated k_i_surge to: {param.value}")
            elif param.name == 'k_d_surge':
                self.config['k_d_surge'] = param.value
                self.get_logger().info(f"Updated k_d_surge to: {param.value}")
            elif param.name == 'k_p_sway':
                self.config['k_p_sway'] = param.value
                self.get_logger().info(f"Updated k_p_sway to: {param.value}")
            elif param.name == 'k_i_sway':
                self.config['k_i_sway'] = param.value
                self.get_logger().info(f"Updated k_i_sway to: {param.value}")
            elif param.name == 'k_d_sway':
                self.config['k_d_sway'] = param.value
                self.get_logger().info(f"Updated k_d_sway to: {param.value}")
            elif param.name == 'pinger_threshold':
                 self.pinger_threshold = param.value
                 self.get_logger().info(f"Updated pinger_threshold to: {param.value}")


        self.update_control_param() # Apply updated PID params

        return SetParametersResult(successful=True)

    def update_control_param(self):
        """Applies the current config parameters to the PID controllers."""
        # Retrieve values from parameters if not already in config
        # self.config['k_p_depth'] = self.get_parameter('k_p_depth').value
        # self.config['k_i_depth'] = self.get_parameter('k_i_depth').value
        # self.config['k_d_depth'] = self.get_parameter('k_d_depth').value

        # self.config['k_p_yaw'] = self.get_parameter('k_p_yaw').value
        # self.config['k_i_yaw'] = self.get_parameter('k_i_yaw').value
        # self.config['k_d_yaw'] = self.get_parameter('k_d_yaw').value

        # self.config['k_p_surge'] = self.get_parameter('k_p_surge').value
        # self.config['k_i_surge'] = self.get_parameter('k_i_surge').value
        # self.config['k_d_surge'] = self.get_parameter('k_d_surge').value

        # self.config['k_p_sway'] = self.get_parameter('k_p_sway').value
        # self.config['k_i_sway'] = self.get_parameter('k_i_sway').value
        # self.config['k_d_sway'] = self.get_parameter('k_d_sway').value

        self.pinger_threshold = self.get_parameter('pinger_threshold').value


        self.pid_depth.reconfig_param(self.config['k_p_depth'], self.config['k_i_depth'], self.config['k_d_depth'])
        self.pid_yaw.reconfig_param(self.config['k_p_yaw'], self.config['k_i_yaw'], self.config['k_d_yaw'])
        self.pid_surge.reconfig_param(self.config['k_p_surge'], self.config['k_i_surge'], self.config['k_d_surge'])
        self.pid_sway.reconfig_param(self.config['k_p_sway'], self.config['k_i_sway'], self.config['k_d_sway'])
        self.get_logger().info("PID and Pinger Threshold parameters updated.")


def main(args=None):
    rclpy.init(args=args)
    node = MyPythonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by keyboard, shutting down.")
        # On shutdown, attempt to disarm and cancel actions
        node.armDisarm(False)
        node.cancel_active_actions_and_reset_modes()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()