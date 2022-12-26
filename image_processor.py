#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
import cv2
import numpy as np
from algorithm.lane_detector import LaneDetector
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.image_pub = self.create_publisher(Image, "/image_proc", 10)
        self.image_sub = self.create_subscription(Image, "/image",self.callback,10)
        self.bridge = CvBridge()

    def callback(self,data):
        try:
            # ROS2의sensor_msgs/Image -> OpenCV의cv::Mat 으로 형변환
            img_cv = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        detector = LaneDetector()
        img_result, x_center = detector.detect(img_cv)
        print(f'center_point: {x_center}')

        #cv2.imshow("Origin Image", img_cv)
        #cv2.imshow("Result Image", img_result)
        #cv2.waitKey(3)

        try:
            # OpenCV의cv::Mat -> ROS2의sensor_msgs/Image 으로 형변환
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_cv, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    try:
        rclpy.init()
        image_processor = ImageProcessor()
        rclpy.spin(image_processor)
        image_processor.destroy_node
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()