import sys

import cv2
# import onnx
# import onnxruntime
import torch
from lane_detector import LaneDetector
import time
import requests
from jetbot import Robot


class Runner:
    def __init__(self, model_path, use_external, external_url):
        self.model_path = model_path
        self.detector = LaneDetector()
        # self.session = onnxruntime.InferenceSession(model_path)
        self.use_external = use_external
        self.external_url = external_url
        self.robot= Robot()

    def inference(self, x):
        if self.use_external:
            return self.inference_external(x)
        else:
            return self.inference_internal(x)

    def inference_internal(self, x):
        # ort_inputs = {self.session.get_inputs()[0].name: x}
        #
        # ort_outs = self.session.run(None, ort_inputs)
        #
        # left_speed = round(float(ort_outs[2][0][0]), 1)
        # right_speed = round(float(ort_outs[2][0][1]), 1)

        # return {"left_speed": left_speed, "right_speed": right_speed}
        return {"left_speed": 0.5, "right_speed": 0.5}

    def inference_external(self, x):
        x_centers = " ".join(str(s) for s in x[0])
        response = requests.post(self.external_url, json={"centers": x_centers})
        return response.json()

    def show_model_info(self):
        pass
    #     onnx_model = onnx.load(self.model_path)
    #     onnx.checker.check_model(onnx_model)
    #
    #     print(onnx_model.graph)
    #
    #     for t in self.session.get_inputs():
    #         print("input:", t.name, t.type, t.shape)
    #
    #     for t in self.session.get_outputs():
    #         print("output:", t.name, t.type, t.shape)


if __name__ == '__main__':
    args = sys.argv

    if len(args) < 2:
        print('Usage: runner.py {} {} {}'.format('{onnx_model_path}', '{external_url}'))
        sys.exit(1)

    # use_external = False if args[2] is None else True
    # external_url = args[2] if use_external else None
    runner = Runner(args[1],
                    False if args[2] is None else True,
                    None if args[2] is None else args[2])
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
        # output = runner.inference_internal(x_center)
        output = runner.inference(x_center)

        left_action = output["left_speed"]
        right_action = output["right_speed"]

        print(f'center_input: {x_center}')
        print(f'left: {output["left_speed"]}, right: {output["right_speed"]}')
        #         runner.robot.left(speed=abs(left_action))
        #         runner.robot.right(speed=abs(right_action)

        runner.robot.set_motors(abs(float(left_action)), abs(float(right_action)))
        time.sleep(3)
        runner.robot.stop()


        # inference(img)
        # ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    if cap.isOpened():
        cap.release()
