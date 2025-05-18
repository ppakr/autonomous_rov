#!/usr/bin/env python

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

class MyPythonNode(Node):
    def __init__(self):
        super().__init__("listenerMIR")
        self.get_logger().info("This node is named listenerMIR")

        self.ns = self.get_namespace()
        self.get_logger().info("namespace =" + self.ns)
        self.pub_msg_override = self.create_publisher(OverrideRCIn, "rc/override", 10)
        self.pub_angle_degree = self.create_publisher(Twist, 'angle_degree', 10)
        self.pub_depth_error = self.create_publisher(Float64, '/depth_error_val', 10)
        self.pub_angular_velocity = self.create_publisher(Twist, 'angular_velocity', 10)
        # self.pub_linear_velocity = self.create_publisher(Twist, 'linear_velocity', 10)
        self.pub_depth_pwm = self.create_publisher(Float64, '/depth_pwm', 10)
        self.yaw_error_pwm = self.create_publisher(Float64, '/yaw_error_pwm', 10)
        # self.filtered_state = self.create_publisher(Odometry, 'filtered_state', 10)
        self.pub_generated_traj = self.create_publisher(Pose, 'generated_traj', 10)
        self.pub_generated_traj_dot = self.create_publisher(Twist, 'generated_traj_dot', 10)


        self.get_logger().info("Publishers created.")

        self.get_logger().info("ask router to create endpoint to enable mavlink/from publication.")

        self.armDisarm(False)  # Not automatically disarmed at startup
        rate = 25  # 25 Hz
        self.setStreamRate(rate)

        self.subscriber()

        # for control
        self.control_rate = 20.0  # Hz
        self.control_period = 1.0 / self.control_rate
        time_tupple = self.get_clock().now().seconds_nanoseconds()
        self.time = time_tupple[0] + (time_tupple[1] * 10**-9)

        # set timer if needed
        timer_period = 0.05  # 50 msec - 20 Hz
        self.timer = self.create_timer(self.control_period, self.timer_callback)
        # self.i = 0
        
        # variables
        # mode -> array
        self.set_mode = [0] * 4
        self.set_mode[0] = True  # Mode manual
        self.set_mode[1] = False  # Mode automatic without correction
        self.set_mode[2] = False  # Mode with correction
        self.set_mode[3] = False  # Mode with pinger

        # Conditions
        self.init_a0 = True
        self.init_p0 = True
        self.arming = False

        self.angle_roll_a0 = 0.0
        self.angle_pitch_a0 = 0.0
        self.angle_yaw_a0 = 0.0
        self.depth_wrt_startup = 0
        self.depth_p0 = 0

        self.pinger_confidence = 0
        self.pinger_distance = 0
        self.pinger_prev_error = 0.0
        self.pinger_error_change = 0.0
        self.pinger_error = 0.0
        self.pinger_threshold = 0.75

        self.free_path = False
        self.search_path = True
        self.crab_walk = False


        self.Vmax_mot = 1900
        self.Vmin_mot = 1100

        # corrections for control
        # assume neutral buoyancy + water bottle
        # ~ 1.5 kgf -> 15 N
        
        self.Correction_yaw_pwm = 1500
        self.Correction_depth = 1500 # need to calculate using water bottle + flotability
        self.surge_pwm = 1500
        self.sway_pwm = 1500

        # controller parameters
        self.config = {}
        self.pid_depth = PIDController(type='linear')
        self.pid_yaw = PIDController(type='angular')
        self.pid_surge = PIDController(type='linear')
        self.pid_sway = PIDController(type='linear')

        self.declare_and_set_params()

        # create parameter callback
        self.add_on_set_parameters_callback(self.callback_params)

        self.desired_depth = 0.0
        self.desired_yaw = 0.0

        # alpha-beta filter
        # self.depth_filter = AlphaBetaFilter(alpha=0.85, beta=0.005)
        self.yaw_filter = AlphaBetaFilter(alpha=0.85, beta=0.005)

        # Initialize trajectory but do not start
        # self.trajectory = CubicTrajectory(z_init=self.depth_p0, z_final=-0.2)
        self.traj_active = False  # Trajectory state
        self.time_init = None
        self.time_final = None
        # self.desired_depth = 0.0

        # Service to start trajectory
        self.srv = self.create_service(SetBool, 'start_trajectory', self.trajectory_callback)

    def trajectory_callback(self, request, response):
        """
        Service callback: Start or stop trajectory generation based on boolean input.
        """
        if request.data:  # True -> Start trajectory
            self.time_init = self.get_clock().now().seconds_nanoseconds()[0] + \
                             self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
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
        if pid > 0:
            pwm = 1500 - 110 * pid
        else:
            pwm = 1500 - 90 * pid
        if pwm > 1900:
            pwm = 1900
        elif pwm < 1100:
            pwm = 1100
        return float(pwm)
    
    def RelAltCallback(self, data):
        """
        Get depth sensor data from this function
        """
        if (self.init_p0):
            # 1st execution, init
            self.depth_p0 = data
            self.init_p0 = False

        # TODO: 
        # setup depth servo control here
        # ...

        # setup for trajectory control
        # print ("data: ", data.data,"type: ", type(data.data))
        
        # get time now
        time_tupple = self.get_clock().now().seconds_nanoseconds()
        current_time = time_tupple[0] + (time_tupple[1] * 10**-9)

        current_depth = data.data

        # check if trajectory is generated
        if self.traj_active:
            # Get waypoint from trajectory
            self.desired_depth, desired_velocity = self.trajectory.get_waypoint(current_time, self.time_init, self.time_final)

            # Publish waypoint
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

        # depth_control = self.pid_to_pwm(floatability)
        ##########################################
        floatability = 0.305
        
        depth_control = self.pid_depth.calculate_pid(self.desired_depth, current_depth, current_time) - floatability
        pub_error_depth = Float64()
        pub_error_depth.data = depth_control
        self.pub_depth_error.publish(pub_error_depth)

        depth_control = self.pid_to_pwm(-depth_control)

        pub_depth = Float64()
        pub_depth.data = depth_control
        self.pub_depth_pwm.publish(pub_depth)

        # calculate alpha-beta filter for task 9
        # filtered_depth, filtered_depth_dot = self.depth_filter.filter(current_depth, current_time)
        # pub_state = Odometry()
        # pub_state.pose.pose.position.z = filtered_depth
        # pub_state.twist.twist.linear.z = filtered_depth_dot
        # self.filtered_state.publish(pub_state)

        # pid task with the observer
        #######3 uncomment for task 10
        ##############################################
        # z_dot = filtered_depth_dot
        # depth_control = self.pid_depth.calculate_pid(self.desired_depth, current_depth, current_time, z_dot) - floatability
        # pub_error_depth = Float64()
        # pub_error_depth.data = depth_control
        # self.pub_depth_error.publish(pub_error_depth)
        # depth_control = self.pid_to_pwm(-depth_control)
        # pub_depth = Float64()
        # pub_depth.data = depth_control
        # self.pub_depth_pwm.publish(pub_depth)
        ##############################################

        # update Correction_depth
        self.Correction_depth = int(depth_control)

    def OdoCallback(self, data):
        """
        Get imu data from this function
        """
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
        angle.angular.z = angle_wrt_startup[2]

        self.pub_angle_degree.publish(angle)

        # Extraction of angular velocity
        p = angular_velocity.x
        q = angular_velocity.y
        r = angular_velocity.z
        vel = Twist() # angular velocity
        vel.angular.x = p
        vel.angular.y = q
        vel.angular.z = r

        # publish velocity
        self.pub_angular_velocity.publish(vel)

        # TODO: setup pid control
        # Only continue if manual_mode is disabled
        if (self.set_mode[0]):
            return
        
        # self.desired_yaw = angle.angular.z
        # alpha-beta filter
        filtered_angle, filtered_angle_dot = self.yaw_filter.filter(angle.angular.z, current_time)

        # Compute yaw error
        yaw_error = self.desired_yaw - angle.angular.z
        if yaw_error > np.pi:
            yaw_error -= 2.0 * np.pi
        elif yaw_error < -np.pi:
            yaw_error += 2.0 * np.pi

        # trajectory generation
        ############# NOTE: Uncomment trajectory for depth control #############
        if self.traj_active:
            self.desired_yaw, desired_yaw_dot = self.trajectory.get_waypoint(current_time, self.time_init, self.time_final)
            self.get_logger().info(f"Generated Waypoint - Yaw: {self.desired_yaw:.3f}, Yaw_dot: {desired_yaw_dot:.3f}")

            # Publish waypoint
            waypoint_msg = Pose()
            waypoint_msg.position.z = self.desired_depth
            self.pub_generated_traj.publish(waypoint_msg)
            
            waypoint_dot_msg = Twist()
            waypoint_dot_msg.linear.z = desired_yaw_dot
            self.pub_generated_traj_dot.publish(waypoint_dot_msg)


            # Stop trajectory if time exceeds
            if current_time > self.time_final:
                self.traj_active = False
                self.get_logger().info("Trajectory complete.")

        # yaw control
        # yaw_control = self.pid_yaw.calculate_pid(self.desired_yaw, angle.angular.z, current_time)
        yaw_control = self.pid_yaw.calculate_pid(self.desired_yaw, filtered_angle, current_time)


        # Send PWM commands to motors
        # yaw command to be adapted using sensor feedback
        # self.Correction_yaw_pwm = 1500

        correction_yaw_pwm = self.pid_to_pwm(yaw_control)

        pub_error_yaw = Float64()
        pub_error_yaw.data = correction_yaw_pwm
        self.yaw_error_pwm.publish(pub_error_yaw)


        self.Correction_yaw_pwm = int(correction_yaw_pwm)

    def crab_walk_callback(self, data):
        self.crab_walk = data.data

    def pinger_callback(self, data):
        if (self.set_mode[0] or self.set_mode[1] or self.set_mode[2]):
            return
        else:
            """
            Get pinger data from this function
            """
            # self.get_logger().info("Pinger data received.")
            # get time now
            time_tupple = self.get_clock().now().seconds_nanoseconds()
            current_time = time_tupple[0] + (time_tupple[1] * 10**-9)

            # extract pinger data
            self.pinger_distance = data.data[0]
            self.pinger_confidence = data.data[1]

            # self.get_logger().info(f"Distance: {self.pinger_distance:.3f}")

            self.pinger_threshold = 1.0  # threshold for pinger confidence

            # Obstacle detected: closer than safe threshold
            if self.pinger_confidence < 70:
                self.get_logger().info("Obstacle detected by pinger")
                self.surge_pwm = 1500
                self.sway_pwm = 1500
                # self.Correction_yaw_pwm = 1500
            
            else:
                
                if self.pinger_distance < self.pinger_threshold:
                    self.free_path = False
                    self.get_logger().info("Obstacle detected by pinger")

                    # Compute control signal to stop or slow down
                    surge_control = self.pid_surge.calculate_pid(
                        self.pinger_distance, self.pinger_threshold, current_time
                    )
                    self.surge_pwm = self.pid_to_pwm(surge_control)

                    # Compute error derivative
                    self.pinger_error = self.pinger_distance - self.pinger_threshold
                    self.pinger_error_change = self.pinger_error - self.pinger_prev_error
                    self.pinger_prev_error = self.pinger_error

                    if np.abs(self.pinger_error_change) > 0.05 and self.crab_walk:
                        self.surge_control = self.pid_surge.calculate_pid(
                            self.pinger_distance, self.pinger_threshold, current_time
                        )
                        self.surge_pwm = self.pid_to_pwm(surge_control)
                        self.sway_pwm = 1600  # Crab walk

                    # If the distance isn't changing much, assume stuck and start search
                    if abs(self.pinger_error_change) < 0.05:
                        self.surge_pwm = 1500
                        self.sway_pwm = 1500
                        self.search_path = True

                # If currently searching for a path
                if self.search_path:
                    self.get_logger().info("Searching for path")
                    self.surge_pwm = 1500
                    self.sway_pwm = 1500
                    self.desired_yaw += -1  # Slowly rotate to look around

                    # If obstacle is now far enough, resume movement
                    if self.pinger_distance > 2 * self.pinger_threshold:
                        self.free_path = True
                        self.search_path = False
                        self.get_logger().info("Free path detected by pinger")

                # If the path is clear, move forward
                if self.free_path:
                    self.get_logger().info("Free path detected by pinger >>> surging")
                    self.surge_pwm = 1535 # * self.pid_sway.k_p  # Move forward
                    # self.Correction_yaw_pwm = 1500
                    self.sway_pwm = 1500   # Keep lateral movement neutral
                    



    def timer_callback(self):
        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        # self.i += 1

        if self.set_mode[0]:  # commands sent inside joyCallback()
            return
        elif self.set_mode[1]:  # Arbitrary velocity command can be defined here to observe robot's velocity, zero by default
            # self.setOverrideRCIN(1500, 1500, 1500, 1500, 1500, 1500)
            # self.get_logger().info("Setmode[1]")
            return
        elif self.set_mode[2]: # dis mode
            # send commands in correction mode
            self.setOverrideRCIN(1500, 1500, self.Correction_depth, self.Correction_yaw_pwm, 1500, 1500)
            self.get_logger().info("Setmode[2]")
            pass
        elif self.set_mode[3]:  # pinger mode
            self.setOverrideRCIN(1500, 1500, 1500, self.Correction_yaw_pwm, self.surge_pwm, self.sway_pwm)
            # self.get_logger().info("Setmode[3]")
            self.get_logger().info(f"yaw_pwm: {self.Correction_yaw_pwm}, surge_pwm: {self.surge_pwm}, sway_pwm: {self.sway_pwm}")

            pass
        else:  # normally, never reached
            pass

    def armDisarm(self, armed):
        # This functions sends a long command service with 400 code to arm or disarm motors
        if (armed):
            traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
            cli = self.create_client(CommandLong, 'cmd/command')
            result = False
            while not result:
                result = cli.wait_for_service(timeout_sec=4.0)
                self.get_logger().info("arming requested, wait_for_service, timeout, result :" + str(result))
            req = CommandLong.Request()
            req.broadcast = False
            req.command = 400
            req.confirmation = 0
            req.param1 = 1.0
            req.param2 = 0.0
            req.param3 = 0.0
            req.param4 = 0.0
            req.param5 = 0.0
            req.param6 = 0.0
            req.param7 = 0.0
            self.get_logger().info("just before call_async")
            resp = cli.call_async(req)
            self.get_logger().info("just after call_async")
            # rclpy.spin_until_future_complete(self, resp)
            self.get_logger().info("Arming Succeeded")
        else:
            traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
            cli = self.create_client(CommandLong, 'cmd/command')
            result = False
            while not result:
                result = cli.wait_for_service(timeout_sec=4.0)
                self.get_logger().info(
                    "disarming requested, wait_for_service, (False if timeout) result :" + str(result))
            req = CommandLong.Request()
            req.broadcast = False
            req.command = 400
            req.confirmation = 0
            req.param1 = 0.0
            req.param2 = 0.0
            req.param3 = 0.0
            req.param4 = 0.0
            req.param5 = 0.0
            req.param6 = 0.0
            req.param7 = 0.0
            resp = cli.call_async(req)
            # rclpy.spin_until_future_complete(self, resp)
            self.get_logger().info("Disarming Succeeded")

    def manageStabilize(self, stabilized):
        # This functions sends a SetMode command service to stabilize or reset
        if (stabilized):
            traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
            cli = self.create_client(SetMode, 'set_mode')
            result = False
            while not result:
                result = cli.wait_for_service(timeout_sec=4.0)
                self.get_logger().info(
                    "stabilized mode requested, wait_for_service, (False if timeout) result :" + str(result))
            req = SetMode.Request()
            req.base_mode = 0
            req.custom_mode = "0"
            resp = cli.call_async(req)
            # rclpy.spin_until_future_complete(self, resp)
            self.get_logger().info("set mode to STABILIZE Succeeded")

        else:
            traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
            result = False
            cli = self.create_client(SetMode, 'set_mode')
            while not result:
                result = cli.wait_for_service(timeout_sec=4.0)
                self.get_logger().info(
                    "manual mode requested, wait_for_service, (False if timeout) result :" + str(result))
            req = SetMode.Request()
            req.base_mode = 0
            req.custom_mode = "19"
            resp = cli.call_async(req)
            # rclpy.spin_until_future_complete(self, resp)
            self.get_logger().info("set mode to MANUAL Succeeded")

    def setStreamRate(self, rate):
        traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
        cli = self.create_client(StreamRate, 'set_stream_rate')
        result = False
        while not result:
            result = cli.wait_for_service(timeout_sec=4.0)
            self.get_logger().info("stream rate requested, wait_for_service, (False if timeout) result :" + str(result))

        req = StreamRate.Request()
        req.stream_id = 0
        req.message_rate = rate
        req.on_off = True
        resp = cli.call_async(req)
        rclpy.spin_until_future_complete(self, resp)
        self.get_logger().info("set stream rate Succeeded")

    def addEndPoint(self):
        traceback_logger = rclpy.logging.get_logger('node_class_traceback_logger')
        cli = self.create_client(EndpointAdd, 'mavros_router/add_endpoint')
        result = False
        while not result:
            result = cli.wait_for_service(timeout_sec=4.0)
            self.get_logger().info(
                "add endpoint requesRelAltCallbackted, wait_for_service, (False if timeout) result :" + str(result))

        req = EndpointAdd.Request()
        req.url = "udp://@localhost"
        req.type = 1  # TYPE_GCS
        resp = cli.call_async(req)
        rclpy.spin_until_future_complete(self, resp)
        self.get_logger().info("add endpoint rate Succeeded")

    def joyCallback(self, data):
        # Joystick buttons
        btn_arm = data.buttons[7]  # Start button
        btn_disarm = data.buttons[6]  # Back button
        btn_manual_mode = data.buttons[3]  # Y button
        btn_automatic_mode = data.buttons[2]  # X button
        btn_corrected_mode = data.buttons[0]  # A button
        btn_collision_mode = data.buttons[1]  # B button

        # Disarming when Back button is pressed
        if (btn_disarm == 1 and self.arming == True):
            self.arming = False
            self.armDisarm(self.arming)

        # Arming when Start button is pressed
        if (btn_arm == 1 and self.arming == False):
            self.arming = True
            self.armDisarm(self.arming)

        # Switch manual, auto anset_moded correction mode
        if (btn_manual_mode and not self.set_mode[0]):
            self.set_mode[0] = True
            self.set_mode[1] = False
            self.set_mode[2] = False
            self.set_mode[3] = False
            self.get_logger().info("Mode manual")
        if (btn_automatic_mode and not self.set_mode[1]):
            self.set_mode[0] = False
            self.set_mode[1] = True
            self.set_mode[2] = False
            self.set_mode[3] = False
            self.get_logger().info("Mode automatic")
        if (btn_corrected_mode and not self.set_mode[2]):
            self.init_a0 = True
            self.init_p0 = True
            # set sum errors to 0 here, ex: Sum_Errors_Vel = [0]*3
            self.set_mode[0] = False
            self.set_mode[1] = False
            self.set_mode[2] = True
            self.set_mode[3] = False
            self.get_logger().info("Mode correction")
        if (btn_collision_mode and not self.set_mode[3]):
            self.set_mode[0] = False
            self.set_mode[1] = False
            self.set_mode[2] = False
            self.set_mode[3] = True
            self.get_logger().info("Mode collision avoidance")
        

    def velCallback(self, cmd_vel):
        # Only continue if manual_mode is enabled
        if (self.set_mode[1] or self.set_mode[2]):
            return
        else:
            self.get_logger().info("Sending...")

        # # creat cmd vel vector with 6 values
        # cmd_vel.linear.x = 0.1
        # cmd_vel.linear.y = 0
        # cmd_vel.linear.z = 0
        # cmd_vel.angular.x = 0
        # cmd_vel.angular.y = 0
        # cmd_vel.angular.z = 0
        
        # Extract cmd_vel message
        roll_left_right = self.mapValueScalSat(cmd_vel.angular.x)
        yaw_left_right = self.mapValueScalSat(-cmd_vel.angular.z)
        ascend_descend = self.mapValueScalSat(cmd_vel.linear.z)
        forward_reverse = self.mapValueScalSat(cmd_vel.linear.x)
        lateral_left_right = self.mapValueScalSat(-cmd_vel.linear.y)
        pitch_left_right = self.mapValueScalSat(cmd_vel.angular.y)

        self.setOverrideRCIN(pitch_left_right, roll_left_right, ascend_descend, yaw_left_right, forward_reverse,
                             lateral_left_right)
    
    def visual_tracker_callback(self, data):
        if (self.set_mode[2] or self.set_mode[0]):
            return
        else:
            self.get_logger().info("Visual tracker data rece ived.")
            roll_left_right = self.mapValueScalSat(data.angular.x)
            yaw_left_right = self.mapValueScalSat(data.angular.z)
            ascend_descend = self.mapValueScalSat(data.linear.z)
            forward_reverse = self.mapValueScalSat(data.linear.x)
            lateral_left_right = self.mapValueScalSat(data.linear.y)
            pitch_left_right = self.mapValueScalSat(data.angular.y)
            # pitch_left_right = 1500
            # roll_left_right = 1500
            # ascend_descend = 1500
            # lateral_left_right = 1500
            # forward_reverse = 1500
            # yaw_left_right = 1500
            yaw_left_right = min(1600, max(1400, yaw_left_right))  # Saturate yaw command
            lateral_left_right = min(1600, max(1400, lateral_left_right))  # Saturate lateral command
            self.setOverrideRCIN(pitch_left_right, roll_left_right, ascend_descend, yaw_left_right, forward_reverse,
                                 lateral_left_right)
        # else:
        #     self.get_logger().info("Not in corrected mode, ignoring visual tracker data.")

    def setOverrideRCIN(self, channel_pitch, channel_roll, channel_throttle, channel_yaw, channel_forward,
                        channel_lateral):
        # This function replaces setservo for motor commands.
        # It overrides Rc channels inputs and simulates motor controls.
        # In this case, each channel manages a group of motors not individually as servo set

        msg_override = OverrideRCIn()
        msg_override.channels[0] = np.uint(channel_pitch)  # pulseCmd[4]--> pitch
        msg_override.channels[1] = np.uint(channel_roll)  # pulseCmd[3]--> roll
        msg_override.channels[2] = np.uint(channel_throttle)  # pulseCmd[2]--> heave
        msg_override.channels[3] = np.uint(channel_yaw)  # pulseCmd[5]--> yaw
        msg_override.channels[4] = np.uint(channel_forward)  # pulseCmd[0]--> surge
        msg_override.channels[5] = np.uint(channel_lateral)  # pulseCmd[1]--> sway
        msg_override.channels[6] = 1500
        msg_override.channels[7] = 1500

        self.pub_msg_override.publish(msg_override)

    def mapValueScalSat(self, value):
        # Correction_Vel and joy between -1 et 1
        # scaling for publishing with setOverrideRCIN values between 1100 and 1900
        # neutral point is 1500
        pulse_width = value * 400 + 1500

        # Saturation
        if pulse_width > 1900:
            pulse_width = 1900
        if pulse_width < 1100:
            pulse_width = 1100

        return int(pulse_width)

    def subscriber(self):
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subjoy = self.create_subscription(Joy, "joy", self.joyCallback, qos_profile=qos_profile)
        self.subjoy  # prevent unused variable warning
        self.subcmdvel = self.create_subscription(Twist, "cmd_vel", self.velCallback, qos_profile=qos_profile)
        self.subcmdvel  # prevent unused variable warning
        self.subimu = self.create_subscription(Imu, "imu/data", self.OdoCallback, qos_profile=qos_profile)
        self.subimu  # prevent unused variable warning

        # visual tracker callback
        self.subvisual_tracker = self.create_subscription(Twist, "visual_tracker", self.visual_tracker_callback,
                                                           qos_profile=qos_profile)
        self.subvisual_tracker  # prevent unused variable warning

        self.subrel_alt = self.create_subscription(Float64, "global_position/rel_alt", self.RelAltCallback,
                                                   qos_profile=qos_profile)
        self.subrel_alt  # prevent unused variable warning

        self.sublaser = self.create_subscription(Float64MultiArray, '/bluerov2/ping1d/data', self.pinger_callback,
                                                 qos_profile=qos_profile)
        self.sublaser

        #crab walk 
        self.subcrab_walk = self.create_subscription(Bool, '/crab_walk', self.crab_walk_callback,
                                                     qos_profile=qos_profile)
        self.subcrab_walk


        self.get_logger().info("Subscriptions done.")


    ### For PID Controller ###

    def update_control_param(self):
        self.pid_depth.reconfig_param(self.config['k_p_depth'], self.config['k_i_depth'], self.config['k_d_depth'])
        self.pid_yaw.reconfig_param(self.config['k_p_yaw'], self.config['k_i_yaw'], self.config['k_d_yaw'])
        self.pid_surge.reconfig_param(self.config['k_p_surge'], self.config['k_i_surge'], self.config['k_d_surge'])
        self.pid_sway.reconfig_param(self.config['k_p_sway'], self.config['k_i_sway'], self.config['k_d_sway'])

    def callback_params(self, params):
        for param in params:
            self.config[param.name] = param.value
        self.update_control_param()
        return SetParametersResult(successful=True)

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

    def declare_and_set_params(self):
        # self.config = {}
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

 
        self.update_control_param()


def main(args=None):
    rclpy.init(args=args)
    node = MyPythonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
