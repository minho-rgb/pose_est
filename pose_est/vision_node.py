import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import math

# [설정]
TARGET_DIST = 0.40
GAIN_PAN = 0.05
GAIN_LINEAR = 0.08
GAIN_DIST = 0.10
DEADZONE = 0.02

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        # 토픽 발행자 생성 (큐 사이즈 10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/dsr01/robot_cmd', 10)
        self.timer = self.create_timer(0.03, self.timer_callback) # 30FPS

        # RealSense & MediaPipe 초기화
        self.init_realsense()
        self.init_mediapipe()
        print(">>> Vision Node Started")

    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def get_finger_status(self, lm):
        fingers = []
        tips = [4, 8, 12, 16, 20]
        # 엄지
        if self.get_distance(lm[4], lm[17]) > self.get_distance(lm[3], lm[17]):
            fingers.append(1)
        else:
            fingers.append(0)
        # 나머지
        for tip in tips[1:]:
            if self.get_distance(lm[0], lm[tip]) > self.get_distance(lm[0], lm[tip-2]):
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        if not color_frame or not depth_frame: return

        img = np.asanyarray(color_frame.get_data())
        depth_map = np.asanyarray(depth_frame.get_data())
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        step_x, step_z, step_ry = 0.0, 0.0, 0.0
        mode_id = 0 # 0:Stop, 1:X, 2:Ry, 3:Full

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = self.get_finger_status(hand_landmarks)
                
                # 추적 좌표 (검지 관절)
                lm = hand_landmarks.landmark[5]
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if 0 <= cx < w and 0 <= cy < h:
                    dist_m = depth_map[cy, cx] * self.depth_scale
                    
                    if dist_m > 0:
                        # 오차 계산
                        err_x = cx - 320
                        err_y = cy - 240
                        err_dist = dist_m - TARGET_DIST

                        raw_ry = err_x * GAIN_PAN * -1
                        raw_x  = -err_y * GAIN_LINEAR
                        raw_z  = err_dist * 1000.0 * GAIN_DIST

                        # 제스처 로직
                        if sum(fingers[1:]) == 0:     # 주먹
                            mode_id = 0
                        elif fingers[1]==1 and sum(fingers[2:])==0: # 검지
                            mode_id = 1
                            step_x = raw_x
                        elif fingers[1]==1 and fingers[2]==1 and sum(fingers[3:])==0: # 검지+중지
                            mode_id = 2
                            step_ry = raw_ry
                        elif sum(fingers[1:]) >= 3:   # 보자기
                            mode_id = 3
                            step_x, step_ry, step_z = raw_x, raw_ry, raw_z

                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        # 메시지 전송 [x, y, z, r, mode]
        msg = Float32MultiArray()
        msg.data = [float(step_x), 0.0, float(step_z), 0.0, float(step_ry), 0.0, float(mode_id)]
        self.publisher_.publish(msg)

        # 화면 표시
        cv2.imshow("Vision Node", img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()