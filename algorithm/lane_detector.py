#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        self.h_min = 0      # hue(색상) 최소값
        self.h_max = 180    # hue(색상) 최대값
        self.s_min = 0      # saturation(채도) 최소값
        self.s_max = 224    # saturation(채도) 최대값
        self.v_min = 10     # value(명도) 최소값
        self.v_max = 100    # value(명도) 최대값

        self.warp_point_top_y = 400             # 워핑할 이미지의 상단 y 좌표
        self.warp_point_bottom_y = 450          # 워핑할 이미지의 하단 y 좌표
        self.warp_point_left_top_x = 25         # 워핑할 이미지의 왼쪽 상단 x 좌표
        self.warp_point_left_bottom_x = 0       # 워핑할 이미지의 왼쪽 하단 x 좌표
        self.warp_point_right_bottom_x = 640    # 워핑할 이미지의 오른쪽 하단 x 좌표
        self.warp_point_right_top_x = 620       # 워핑할 이미지의 오른쪽 상단 x 좌표

        self.points = (np.array([[self.warp_point_left_top_x, self.warp_point_top_y],
                                 [self.warp_point_right_top_x, self.warp_point_top_y],
                                 [self.warp_point_left_bottom_x, self.warp_point_bottom_y],
                                 [self.warp_point_right_bottom_x, self.warp_point_bottom_y]]))

        self.lane_window_name = "LaneWindow"
        self.sliding_window_name = "SlidingWindow"

    def warp_img(self, in_img, points, w, h):
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(in_img, matrix, (w, h))

        return img_warp

    def filter_img(self, in_img):
        img_filtered = cv2.Canny(in_img, 50, 100)

        img_sobel_x = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=3)
        img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

        img_sobel_y = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=3)
        img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

        img_filtered = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 2, 0);

        kernel = np.ones((3, 3), np.uint8)
        img_filtered = cv2.dilate(img_filtered, kernel, iterations = 1)

        return img_filtered

    def filter_img_hsv(self, in_img, h_min, h_max, s_min, s_max, v_min, v_max):
        img_hsv = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([h_min, s_min, v_min])
        upper_white = np.array([h_max, s_max, v_max])
        img_filtered = cv2.inRange(img_hsv, lower_white, upper_white)
        img_filtered = cv2.GaussianBlur(img_filtered, (3, 3), 0)

        return img_filtered

    def get_hist(self, in_img, img_y_top=0):
        hist = np.sum(in_img[img_y_top:, :], axis=0)
        return hist

    def sliding_window(self, in_img, num_windows=1, margin=50, minpix=1, draw_windows=True):
        global x_center, x_center_gap
        center_margin = 40
        out_img = np.dstack((in_img, in_img, in_img))
        histogram = self.get_hist(in_img)

        # find peaks of left and right halves
        first_occur_idx = np.where(histogram > 0)[0][0]
        # x_base = np.argmax(histogram)
        x_base = first_occur_idx

        window_height = int((in_img.shape[0] / num_windows))

        # Identify the x and y positions of all nonzero_indices pixels in the image
        nonzero_inds = in_img.nonzero()
        nonzero_y = np.array(nonzero_inds[0])
        nonzero_x = np.array(nonzero_inds[1])

        # Current positions to be updated for each window
        x_current = x_base

        # 좌/우측 레인의 픽셀값 리스트
        lane_inds = []

        # Step through the windows one by one
        for num_window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = in_img.shape[0] - (num_window + 1) * window_height
            win_y_high = in_img.shape[0] - num_window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # draw window
            if draw_windows:
                cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),
                              (100,255,255), 3)
                cv2.putText(out_img, str(num_window), (win_x_high-40, win_y_low+((win_y_high-win_y_low)//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # window 범위 내에 있는 nonzero_indices pixel 추출
            left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                         (nonzero_x >= win_x_low) &  (nonzero_x < win_x_high)).nonzero()[0]

            # 추출한 픽셀 인덱스 추가
            lane_inds.append(left_inds)

            # found > minpix pixels 이면 다음 윈도우의 중심점 평균 재계산
            if len(left_inds) > minpix:
                x_current = int(np.mean(nonzero_x[left_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # 좌/우 레인의 픽셀 위치 추출
        x = nonzero_x[lane_inds]
        left_y = nonzero_y[lane_inds]

        ploty = np.linspace(0, in_img.shape[0] - 1, in_img.shape[0])
        left_fitx = np.zeros(in_img.shape[0])

        left_fit = np.zeros(3)
        x_center = center_margin

        if len(x) > 0:
            # 2차함수 근사
            # left_fit = np.polyfit(left_y, x, 2)
            # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

            # 1차함수 근사
            left_fit = np.polyfit(left_y, x, 1)
            left_fitx = left_fit[0]*ploty + left_fit[1]

            out_img[nonzero_y[lane_inds], nonzero_x[lane_inds]] = [0, 0, 255]
            x_center = left_fitx + center_margin

        return out_img, left_fitx, left_fit, x_center

    def detect(self, in_img, draw_windows=True):
        img_filtered = self.filter_img_hsv(in_img, self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max)
        img_filtered = self.filter_img(img_filtered)
        h, w = img_filtered.shape
        img_warp_filtered = self.warp_img(img_filtered, self.points, w, h-self.warp_point_top_y)
        img_lane, lane, lane_poly, x_center = self.sliding_window(img_warp_filtered)

        for i in range(x_center.size):
            cv2.circle(in_img, (int(x_center[i]), i + self.warp_point_top_y), 2, (255, 0, 0), -1)

        if draw_windows:
            cv2.moveWindow(self.sliding_window_name, in_img.shape[1], 0)
            cv2.imshow(self.lane_window_name, in_img)
            cv2.imshow(self.sliding_window_name, img_lane)

        return in_img, x_center


if __name__ == '__main__':
    args = sys.argv

    if len(args) != 2:
        print('Usage: lane_detector.py {}'.format('{true|false}'))
        sys.exit(1)

    try:
        detector = LaneDetector()
        # cap = cv2.VideoCapture('./inputs/input.mp4')
        cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
        frame_counter = 0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            # 카메라 프레임 읽기
            success, img = cap.read()
            img = cv2.resize(img, (frame_width, frame_height))
            img_detect, x_center = detector.detect(img, True if args[1] == 'true' else False)
            if not args[1]:
                print(x_center)

            # ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        if cap.isOpened():
            cap.release()

    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
