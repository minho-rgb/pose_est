import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import mediapipe as mp
import math

# [설정]
# 웹캠용 거리 추정 기준 (손목~중지 뿌리 픽셀 거리)
# 이 값보다 손이 커지면(가까우면) 뒤로, 작으면(멀면) 앞으로 이동합니다.
TARGET_HAND_SIZE = 100.0  

GAIN_PAN = 0.05
GAIN_LINEAR = 0.08
GAIN_DIST = 0.15     # 거리 제어 게인 (Webcam용 조절)
DEADZONE = 0.02

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        # 토픽 발행자 생성
        self.publisher_ = self.create_publisher(Float32MultiArray, '/dsr01/robot_cmd', 10)
        self.timer = self.create_timer(0.03, self.timer_callback) # 30FPS

        # 웹캠 초기화 (0번 장치)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: 웹캠을 열 수 없습니다.")
        else:
            # 해상도 설정 (640x480)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # MediaPipe 초기화
        self.init_mediapipe()
        print(">>> Vision Node (Webcam Mode) Started")

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
        # 나머지 (검지~소지)
        for tip in tips[1:]:
            if self.get_distance(lm[0], lm[tip]) > self.get_distance(lm[0], lm[tip-2]):
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 웹캠은 좌우 반전이 필요할 수 있음 (거울 모드)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        step_x, step_z, step_ry = 0.0, 0.0, 0.0
        mode_id = 0 # 0:Stop, 1:X, 2:Ry, 3:Full

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = self.get_finger_status(hand_landmarks)
                
                # [1] 추적 좌표 (검지 관절, 5번)
                lm = hand_landmarks.landmark[5]
                cx, cy = int(lm.x * w), int(lm.y * h)

                # [2] 가상 거리 측정 (손목 0번 ~ 중지 뿌리 9번 사이의 픽셀 거리)
                # 손이 가까우면 이 거리가 커지고, 멀면 작아짐
                lm_wrist = hand_landmarks.landmark[0]
                lm_mid_root = hand_landmarks.landmark[9]
                
                # 픽셀 단위 거리 계산
                curr_hand_size = math.hypot(
                    (lm_wrist.x - lm_mid_root.x) * w, 
                    (lm_wrist.y - lm_mid_root.y) * h
                )

                # 오차 계산
                err_x = cx - 320         # 화면 중앙(320)에서의 좌우 오차
                err_y = cy - 240         # 화면 중앙(240)에서의 상하 오차
                
                # 거리 오차 (기준 크기 - 현재 크기)
                # 현재 크기가 150이고 기준이 100이면(너무 가까움) -> 양수 -> 뒤로 가야 함
                # 현재 크기가 50이고 기준이 100이면(너무 멂) -> 음수 -> 앞으로 가야 함
                # 로봇 좌표계 방향에 맞춰 부호 조정 필요 (여기선 단순 비례 제어)
                err_dist = (curr_hand_size - TARGET_HAND_SIZE) * 0.01 

                # 로봇 이동량 계산 (Raw Step)
                raw_ry = err_x * GAIN_PAN * -1       # 좌우 -> Ry 회전
                raw_x  = -err_y * GAIN_LINEAR        # 상하 -> X축 수직 이동
                raw_z  = -err_dist * 1000.0 * GAIN_DIST # 거리 -> Z축 전후 이동 (부호 반전 주의)

                # [3] 제스처 로직 (모드 결정)
                if sum(fingers[1:]) == 0:     # 주먹 (정지)
                    mode_id = 0
                    cv2.putText(frame, "STOP", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                elif fingers[1]==1 and sum(fingers[2:])==0: # 검지 (상하 X축)
                    mode_id = 1
                    step_x = raw_x
                    cv2.putText(frame, "X-Axis Only", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                elif fingers[1]==1 and fingers[2]==1 and sum(fingers[3:])==0: # 가위 (좌우 Ry축)
                    mode_id = 2
                    step_ry = raw_ry
                    cv2.putText(frame, "Ry-Axis Only", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                elif sum(fingers[1:]) >= 3:   # 보자기 (전체 추적)
                    mode_id = 3
                    step_x, step_ry, step_z = raw_x, raw_ry, raw_z
                    cv2.putText(frame, "Full Tracking", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 화면 표시
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # 거리 정보 디버깅 표시
                msg_dist = f"Size: {int(curr_hand_size)} (Target: {int(TARGET_HAND_SIZE)})"
                cv2.putText(frame, msg_dist, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 메시지 전송 [x, y, z, r, mode]
        msg = Float32MultiArray()
        msg.data = [float(step_x), 0.0, float(step_z), 0.0, float(step_ry), 0.0, float(mode_id)]
        self.publisher_.publish(msg)

        # 화면 표시
        cv2.imshow("Webcam Vision Node", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()