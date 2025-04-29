class VerticalAlignmentTask(FinalBehaviorMainTask):
    def __init__(self, timeout=2):
        self._bridge = CvBridge()
        self._error_last = 0
        self._error = 0

    def onStart(self, dtros):
        vehicle_name = os.environ["VEHICLE_NAME"]
        self._raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)
    
    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistort = ImageOperations.undistort(image)
        homography = ImageOperations.getHomography(undistort)
        lines = ImageOperations.getImageLines(homography)

        if lines is not None:
            angles = []

            for line in lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta)
                angles.append(angle_deg)

                # Convert polar coords to cartesian and draw line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(homography, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines


            avg_angle = sum(angles) / len(angles)
            avg_verticality = 0 if 100 - min(abs(avg_angle), abs(180 - avg_angle)) <= 0 else 100 - min(abs(avg_angle), abs(180 - avg_angle))

            if avg_angle > 90:
                self._error = 100 - avg_verticality
            else:
                self._error = -1*(100 - avg_verticality)

            print(f"Average Angle: {avg_angle:.2f}°")
            print(f"Average Verticality Score: {avg_verticality:.2f}")
        
        cv2.imshow("Edges", homography)
        cv2.waitKey(1)
    
    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            _, error_last_updated, message = PIDOperations.getRotatePIDWheelMsg(error_last=self._error_last, integration_stored=0, error=-1*self._error, proportional_gain=0.005, derivative_gain=0.005, integral_gain=0, integral_saturation=0)
            
            self._error_last = error_last_updated

            dtros._wheels_publisher.publish(message)

            rate.sleep()

class RotateRightToTagTask(FinalBehaviorMainTask):
    def __init__(self,target_stall, precision=40, tolerance=0.04, radians=math.pi/2, angular_velocity=1, R=0):
        super().__init__()
        self._precision = precision
        self._tolerance = tolerance
        self._radians = radians
        self._angular_velocity=angular_velocity
        self._R = R

        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0

        self._resolution = 135
        self._l = 0.050

        self._target_stall = target_stall
        self.detector = dt_apriltags.Detector(families="tag36h11")

        self._target_tag_error = math.inf
        self._bridge = CvBridge()

    def onStart(self, dtros):

        vehicle_name = os.environ["VEHICLE_NAME"]

        self._radius = rospy.get_param(f'/{vehicle_name}/kinematics_node/radius', 0.0318)
        
        left_encoder_topic = f"/{vehicle_name}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"/{vehicle_name}/right_wheel_encoder_node/tick"

        dtros._sub_left_wheel = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.callback_left_wheel)
        dtros._sub_right_wheel = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.callback_right_wheel)

        self._raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)

    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistort = ImageOperations.undistort(image)
        undistort_gray = ImageOperations.getGrayscale(undistort)

        # Convert JPEG bytes to CV image
        image = undistort_gray

        height, width = image.shape[:2]

        # If the image is grayscale (single channel), convert it to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Detect AprilTags (assuming detector works on grayscale, you might need to convert back or detect on the original)
        results = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        for r in results:
            if r.tag_id == self._target_stall.value:
                self._target_tag_error = ImageOperations.getErrorFromCenterInAxisX(image, r.center[0])
                return
        
        self._target_tag_error = math.inf


    def callback_left_wheel(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
    
    def callback_right_wheel(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data

    def runTask(self, dtros):
        
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R

        msg = f""" Running a curve task ... target_angle : {target_radian}, R: {R}, precision: {precision}, tolerance: {tolerance} """
        rospy.loginfo(msg)

        rate = rospy.Rate(precision)

        v_r = (R - self._l*3) * angular_velocity
        v_l = (R + self._l*3) * angular_velocity

        message = WheelsCmdStamped(vel_left=v_l, vel_right=v_r)

        while not rospy.is_shutdown():
            print(self._target_tag_error)

            if abs(self._target_tag_error) < 10: 
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            dtros._wheels_publisher.publish(message)
            rate.sleep()

class VehicleAvoidance(FinalBehaviorMainTask):
    def __init__(self): 
        self._lane_correction_delay = 2
        self._target_lane = Lane.RIGHT
        self._duckie_detected_time_stamp = None

        self.duckie_detection_sensitivity = 1000
        self.duckie_detection_distance = 15000

        self._error_last = 0
        self._error = 0

        self._integration_stored = 0
        self._bridge = CvBridge()
        
    def onStart(self, dtros):
        vehicle_name = os.environ["VEHICLE_NAME"]
        self._raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        dtros._sub_raw_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)

    def callback_raw_image(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistorted = ImageOperations.undistort(image)
        homography = ImageOperations.getHomography(undistorted)
        mask_white = ImageOperations.getWhiteMask(homography)
        mask_yellow = ImageOperations.getYellowMask(homography)
        mask_blue = ImageOperations.getBlueMask(undistorted)

        if self.isDuckiebotBotVisible(mask_blue):
            self._duckie_detected_time_stamp = time.time()
        
        if self._duckie_detected_time_stamp is not None and (time.time() - self._duckie_detected_time_stamp) < self._lane_correction_delay:
            self._target_lane = Lane.LEFT
        else:
            self._target_lane = Lane.RIGHT

        if self._target_lane == Lane.LEFT:
            _, width = mask_yellow.shape[:2]
            mask_yellow[:, :90] = 0
            yellow_error = MaskOperations.computeErrorInAxisX(mask=mask_yellow, target_x=489, pixel_value=1)
            white_error = MaskOperations.computeErrorInAxisX(mask=mask_white, target_x=100, pixel_value=1)
        else:
            yellow_error = MaskOperations.computeErrorInAxisX(mask=mask_yellow, target_x=100, pixel_value=1)
            white_error = MaskOperations.computeErrorInAxisX(mask=mask_white, target_x=489, pixel_value=1)

        if yellow_error > 100000 and white_error < -100000:
            self._error = (yellow_error + abs(white_error)) 
        elif yellow_error < -100000 and white_error > 100000:
            self._error = (yellow_error + -1*white_error) 
        else:
            self._error = yellow_error + white_error

        cv2.imshow("White Mask", mask_white)
        cv2.imshow("Yellow Mask", mask_yellow)
        cv2.waitKey(1)

    def isDuckiebotBotVisible(self, undistort_blue_mask):
        _, x_coords = np.where(undistort_blue_mask > 0)

        if len(x_coords) == 0:
            return False
        
        x_variance = np.var(x_coords)
        magnitude = len(x_coords)

        if (magnitude >= self.duckie_detection_sensitivity):
            if(x_variance <= self.duckie_detection_distance):
                return True
        return False


    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            integration_stored_updated, error_last_updated, message = PIDOperations.getForwardPIDWheelMsg(base_velocity=0.3, error_last=self._error_last, integration_stored=self._integration_stored, error=self._error, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, integral_saturation=500000)
            self._error_last = error_last_updated
            self._integration_stored = integration_stored_updated
            dtros._wheels_publisher.publish(message)
            rate.sleep()