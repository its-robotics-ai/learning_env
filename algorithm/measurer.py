#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import cv2
import numpy as np


class Measurer:
    measure_type = None

    h_min = 0      # hue(색상) 최소값
    h_max = 180    # hue(색상) 최대값
    s_min = 0      # saturation(채도) 최소값
    s_max = 224    # saturation(채도) 최대값
    v_min = 10     # value(명도) 최소값
    v_max = 100    # value(명도) 최대값
    trackbar_window_name = "Original"
    mask_window_name = "Mask"
    result_window_name = "Result"

    warp_point_top_y = 400             # 워핑할 이미지의 상단 y 좌표
    warp_point_bottom_y = 450          # 워핑할 이미지의 하단 y 좌표
    warp_point_left_top_x = 25         # 워핑할 이미지의 왼쪽 상단 x 좌표
    warp_point_left_bottom_x = 0       # 워핑할 이미지의 왼쪽 하단 x 좌표
    warp_point_right_bottom_x = 640    # 워핑할 이미지의 오른쪽 하단 x 좌표
    warp_point_right_top_x = 620       # 워핑할 이미지의 오른쪽 상단 x 좌표
    warp_initial_vals = [100, 100, 100, 300]   # 워핑 초기값

    def __init__(self, in_measure_type):
        self.measure_type = in_measure_type

    def measure_hsv(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", self.trackbar_window_name)
        h_max = cv2.getTrackbarPos("Hue Max", self.trackbar_window_name)
        s_min = cv2.getTrackbarPos("Sat Min", self.trackbar_window_name)
        s_max = cv2.getTrackbarPos("Sat Max", self.trackbar_window_name)
        v_min = cv2.getTrackbarPos("Val Min", self.trackbar_window_name)
        v_max = cv2.getTrackbarPos("Val Max", self.trackbar_window_name)
        print(f'h_min: {h_min}, h_max: {h_max}, s_min: {s_min}, s_max: {s_max}, v_min: {v_min}, v_max: {v_max}')
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        cv2.moveWindow(self.mask_window_name, img.shape[1], 0)
        cv2.moveWindow(self.result_window_name, img.shape[1]*2, 0)
        cv2.imshow(self.trackbar_window_name, img)
        cv2.imshow(self.mask_window_name, mask)
        cv2.imshow(self.result_window_name, imgResult)

    def measure_warp_point(self, img):
        h, w, c = img.shape

        width_top = cv2.getTrackbarPos('Top Width', self.trackbar_window_name)
        height_top = cv2.getTrackbarPos('Top Height', self.trackbar_window_name)
        width_bottom = cv2.getTrackbarPos('Bottom Width', self.trackbar_window_name)
        height_bottom = cv2.getTrackbarPos('Bottom Height', self.trackbar_window_name)

        points = np.float32([(width_top, height_top), (w - width_top, height_top),
                             (width_bottom, height_bottom), (w - width_bottom, height_bottom)])

        print(f'left_top: {points[0]}, right_top: {points[1]}, left_bottom: {points[2]}, right_bottom: {points[3]}')

        for x in range(len(points)):
            cv2.circle(img, (int(points[x][0]), int(points[x][1])), 3, (0, 0, 255), cv2.FILLED)

        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (w, h))

        cv2.imshow(self.trackbar_window_name, img)
        cv2.imshow(self.result_window_name, img_warp)

    def initialize_hsv_trackbars(self):
        cv2.namedWindow(self.trackbar_window_name)
        cv2.createTrackbar("Hue Min", self.trackbar_window_name, self.h_min, 360, self.nothing)  # hue(색상) 최소값
        cv2.createTrackbar("Hue Max", self.trackbar_window_name, self.h_max, 360, self.nothing)  # hue(색상) 최대값
        cv2.createTrackbar("Sat Min", self.trackbar_window_name, self.s_min, 255, self.nothing)  # saturation(채도) 최소값
        cv2.createTrackbar("Sat Max", self.trackbar_window_name, self.s_max, 255, self.nothing)  # saturation(채도) 최대값
        cv2.createTrackbar("Val Min", self.trackbar_window_name, self.v_min, 255, self.nothing)  # value(명도) 최소값
        cv2.createTrackbar("Val Max", self.trackbar_window_name, self.v_max, 255, self.nothing)  # value(명도) 최대값

    def initialize_warp_trackbars(self, initial_trackbar_vals, w, h):
        cv2.namedWindow(self.trackbar_window_name)
        cv2.createTrackbar('Top Width', self.trackbar_window_name, initial_trackbar_vals[0], w // 2, self.nothing)
        cv2.createTrackbar('Top Height', self.trackbar_window_name, initial_trackbar_vals[1], h, self.nothing)
        cv2.createTrackbar('Bottom Width', self.trackbar_window_name, initial_trackbar_vals[2], w // 2, self.nothing)
        cv2.createTrackbar('Bottom Height', self.trackbar_window_name, initial_trackbar_vals[3], h, self.nothing)

    def nothing(self, a):
        pass


if __name__ == '__main__':
    args = sys.argv

    if len(args) != 2 or args[1] not in ['hsv', 'warp']:
        print('Usage: measurer.py {} '.format('{hsv|warp}'))
        sys.exit(1)

    measure_type = args[1]

    measurer = Measurer(measure_type)

    #cap = cv2.VideoCapture("inputs/input.mp4") #동영상 파일에서 읽기
    cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
    frame_counter = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if measure_type == 'hsv':
        measurer.initialize_hsv_trackbars()
    elif measure_type == 'warp':
        measurer.initialize_warp_trackbars(measurer.warp_initial_vals, frame_width, frame_height)

    while cap.isOpened():
        # 카메라 프레임 읽기
        success, img = cap.read()

        if measure_type == 'hsv':
            measurer.measure_hsv(img)
        elif measure_type == 'warp':
            measurer.measure_warp_point(img)

        # ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()