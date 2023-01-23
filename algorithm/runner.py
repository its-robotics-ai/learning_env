import sys

import cv2
import onnx
import onnxruntime
import torch
from lane_detector import LaneDetector
import time
# from jetbot import Robot


class Runner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.detector = LaneDetector()
        self.session = onnxruntime.InferenceSession(model_path)

    def inference(self, x):
        ort_inputs = {self.session.get_inputs()[0].name: x}
        ort_outs = self.session.run(None, ort_inputs)
        return ort_outs

    def show_model_info(self):
        onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(onnx_model)

        print(onnx_model.graph)

        for t in self.session.get_inputs():
            print("input:", t.name, t.type, t.shape)

        for t in self.session.get_outputs():
            print("output:", t.name, t.type, t.shape)


if __name__ == '__main__':
    args = sys.argv

    if len(args) != 2:
        print('Usage: runner.py {}'.format('{onnx_model_path}'))
        sys.exit(1)

    runner = Runner(args[1])
    detector = LaneDetector()
    cap = cv2.VideoCapture('./inputs/input.mp4')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    runner.show_model_info()

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.resize(img, (frame_width, frame_height))
        img_detect, x_center = detector.detect(img, False)
        x_center = x_center[0:8].astype('float32').reshape(1, 8)
        tensor_x_center = torch.Tensor(x_center)
        output = runner.inference(x_center)

        left_action = round(float(output[2][0][0]), 1)
        right_action = round(float(output[2][0][1]), 1)

        print(f'left: {left_action}, right: {right_action}')
        # robot.left(speed=left_action)
        # robot.right(speed=right_action)

        time.sleep(2)


        # inference(img)
        # ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    if cap.isOpened():
        cap.release()
