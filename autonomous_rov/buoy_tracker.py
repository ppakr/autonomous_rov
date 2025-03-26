import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

class BuoyTracker(Node):
    def __init__(self):
        super().__init__('buoy_tracker')

        # Declare parameters with a default list of integers
        self.declare_parameter('lower_hsv', [0, 144, 117]) #buoy red  # Default: [146, 128, 101] # pablo bottle
        self.declare_parameter('upper_hsv', [33, 220, 255]) #buoy red  # Default: [179, 255, 255] # pablo bottle

        # ROS2 subscribers & publishers
        self.subscription = self.create_subscription(
            Image,
            'video_topic',  # This is the topic your camera publisher uses
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Float64MultiArray, 'tracked_point', 10)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Minimum detection size
        self.min_width = 20
        self.min_height = 20

        # Set the initial HSV values from parameters
        self.set_hsv_thresholds()

        # Parameter update callback to check and update the HSV thresholds at runtime
        self.create_timer(1.0, self.update_hsv_values)

        self.get_logger().info("Buoy Tracker Node Initialized.")

    def set_hsv_thresholds(self):
        # Access HSV parameters using .get_parameter_value().get_parameter_value()
        lower_hsv_param = self.get_parameter('lower_hsv').get_parameter_value().integer_array_value
        upper_hsv_param = self.get_parameter('upper_hsv').get_parameter_value().integer_array_value

        # Convert these arrays into numpy arrays for further processing
        self.lower_hsv = np.array(lower_hsv_param, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv_param, dtype=np.uint8)

    def update_hsv_values(self):
        # Update HSV values from parameters at runtime
        self.set_hsv_thresholds()

    def remove_reflections(self, frame, mask):
        """
        Detects the waterline and removes reflections above it.
        The waterline is modeled as a polynomial curve, taking into account camera motion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        edges = cv2.Canny(gray, 50, 150)  # Detect edges

        # Visualize edges to see if waterline is detected
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

        # Detect horizontal lines using Hough Transform or other methods to detect edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            # Find the points of the detected lines (water surface points)
            points = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if y1 == y2:  # Only consider horizontal lines (water surface)
                        points.append((x1, y1))

            if len(points) > 0:
                # Sort points by their y-coordinates (row number)
                points = sorted(points, key=lambda x: x[1])

                # Separate x and y coordinates
                x_points = np.array([p[0] for p in points])
                y_points = np.array([p[1] for p in points])

                # Visualize the detected points on the image (for debugging)
                for p in points:
                    cv2.circle(frame, p, 3, (0, 255, 0), -1)

                if len(x_points) > 5:  # Make sure there are enough points to fit a curve
                    # Fit a polynomial curve to the detected points (waterline)
                    poly_coeffs = np.polyfit(y_points, x_points, deg=2)  # 2nd-degree polynomial (quadratic)

                    # Generate the fitted waterline
                    y_fit = np.linspace(min(y_points), max(y_points), num=500)  # Generate y values for fitting
                    x_fit = np.polyval(poly_coeffs, y_fit)  # Get corresponding x values from the polynomial

                    # Visualize the fitted polynomial curve on the image
                    for i in range(len(y_fit)):
                        cv2.circle(frame, (int(x_fit[i]), int(y_fit[i])), 2, (255, 0, 0), -1)

                    # Convert the fitted points back to integer coordinates
                    x_fit_int = np.array(np.round(x_fit), dtype=int)
                    y_fit_int = np.array(np.round(y_fit), dtype=int)

                    # Create a mask to remove reflections above the fitted curve
                    for i in range(len(x_fit_int)):
                        if y_fit_int[i] < frame.shape[0]:  # Ensure within image bounds
                            mask[y_fit_int[i]:, x_fit_int[i]] = 0  # Mask out the area above the waterline
                else:
                    self.get_logger().warn("Not enough points detected for polynomial fitting.")
            else:
                self.get_logger().warn("No horizontal lines detected for waterline.")
        else:
            self.get_logger().warn("No lines detected by Hough Transform.")

        return mask

    def image_callback(self, msg):
        try:
            
            """
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Mask using predefined HSV range
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            """

            # Convert ROS2 image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Mask using predefined HSV range
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

            # **Apply water reflection removal**
            mask = self.remove_reflections(frame, mask)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            center_x, center_y, area = -1, -1, 0  # Default values
            if contours:
                contour=max(contours,key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)

                if w >= self.min_width and h >= self.min_height:
                    # Calculate the center point
                    center_x, center_y = x + w // 2, y + h // 2
                    area = w * h
                    # Draw rectangle & marker
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.line(frame, (center_x - 10, center_y - 10), (center_x + 10, center_y + 10), (255, 0, 0), 2)
                    cv2.line(frame, (center_x - 10, center_y + 10), (center_x + 10, center_y - 10), (255, 0, 0), 2)

                

            # Publish center coordinates
            msg = Float64MultiArray()
            msg.data = [float(center_x), float(center_y), float(area)]
            self.publisher.publish(msg)

            # Display windows
            cv2.imshow("Buoy Tracking", frame)  # This will show the frame with bounding box
            cv2.imshow("Mask", mask)  # Show the mask to debug how well the buoy is detected
            cv2.waitKey(1)  # Wait for a key event

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BuoyTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # Ensure windows are closed when ROS2 shuts down

if __name__ == '__main__':
    main()
