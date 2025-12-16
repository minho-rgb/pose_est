import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import mediapipe as mp
import math

# =========================================================
# [설정] 이미지 속 Tool 좌표계 기준 매핑
# =========================================================
TARGET_HAND_SIZE = 250.0  # 기준 손 크기

# 1. 반응 속도 (Gain)
GAIN_X = 0.08   # 로봇 X축 민감도
GAIN_Y = 0.08   # 로봇 Y축 민감도
GAIN_Z = 0.12   # 로봇 Z축 민감도

# 2. 방향 설정 (이미지 기준)
DIR_ROBOT_X = 1   
DIR_ROBOT_Y = 1  
DIR_ROBOT_Z = -1   

# 3. 데드존 (떨림 방지)
DEADZONE_PIXEL = 20
DEADZONE_DIST = 10.0

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/dsr01/robot_cmd', 10)
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz 전송

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.init_mediapipe()
        print(">>> Vision Node Started")
        print("    - Fist(0): Grip Close")
        print("    - Index(1): 2D Move")
        print("    - Victory(2): 3D Move")
        print("    - 3 Fingers(3): Stop")
        print("    - All Open(5): Grip Open")

    def init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def get_finger_status(self, hand_landmarks):
        lm = hand_landmarks.landmark 
        fingers = []
        # 엄지
        if self.get_distance(lm[4], lm[17]) > self.get_distance(lm[3], lm[17]):
            fingers.append(1)
        else:
            fingers.append(0)
        # 검지~소지
        tips = [8, 12, 16, 20]
        for tip in tips:
            if self.get_distance(lm[0], lm[tip]) > self.get_distance(lm[0], lm[tip-2]):
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 거울 모드
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # 초기화 [x, y, z, rx, ry, rz]
        step = [0.0] * 6 
        mode_id = 0 # 기본값: 정지

        # 화면 중앙 가이드 (십자선 혹은 원)
        cv2.circle(frame, (320, 240), 5, (0, 255, 255), 2)
        cv2.putText(frame, "Center", (330, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = self.get_finger_status(hand_landmarks)
                total_fingers = sum(fingers) # 펴진 손가락 개수

                # 1. 손 중심 및 크기 계산
                lm = hand_landmarks.landmark[9] 
                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_wrist = hand_landmarks.landmark[0]
                curr_size = math.hypot((lm_wrist.x-lm.x)*w, (lm_wrist.y-lm.y)*h) * 2.5 

                # 2. 오차 계산
                err_cam_y = (240 - cy) 
                err_cam_x = (cx - 320)
                err_cam_z = (curr_size - TARGET_HAND_SIZE)

                # 데드존 적용
                if abs(err_cam_y) < DEADZONE_PIXEL: err_cam_y = 0
                if abs(err_cam_x) < DEADZONE_PIXEL: err_cam_x = 0
                if abs(err_cam_z) < DEADZONE_DIST: err_cam_z = 0

                # 3. 로봇 이동량 계산
                move_x = err_cam_x * GAIN_X * DIR_ROBOT_X
                move_y = err_cam_y * GAIN_Y * DIR_ROBOT_Y
                move_z = err_cam_z * GAIN_Z * DIR_ROBOT_Z

                # 4. 제스처 인식 로직 (수정됨)
                
                # (1) 주먹 ✊ -> 그리퍼 닫기 (Mode 3)
                if total_fingers == 0:
                    mode_id = 3
                    cv2.putText(frame, "GRIP CLOSE (Fist)", (cx, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # (2) 보자기 🖐️ -> 그리퍼 열기 (Mode 4)
                elif total_fingers == 5:
                    mode_id = 4
                    cv2.putText(frame, "GRIP OPEN (All Open)", (cx, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # (3) 손가락 3개 -> 정지 (Mode 0)
                elif total_fingers == 3:
                    mode_id = 0
                    cv2.putText(frame, "STOP (3 Fingers)", (cx, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # (4) 검지 ☝️ -> 2D 이동 (Mode 1)
                elif fingers[1]==1 and total_fingers==1:
                    mode_id = 1
                    step[0], step[1], step[2] = move_x, move_y, 0.0
                    cv2.putText(frame, "2D Move", (cx, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # (5) 가위 ✌️ -> 3D 이동 (Mode 2)
                elif fingers[1]==1 and fingers[2]==1 and total_fingers==2:
                    mode_id = 2
                    step[0], step[1], step[2] = move_x, move_y, move_z
                    cv2.putText(frame, "3D Move", (cx, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ---------------------------------------------------------
                # [복구된 부분] 시각화: 랜드마크 그리기 + 정보 텍스트 + 연결선
                # ---------------------------------------------------------
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 이동 명령 정보 출력
                info = f"RX:{step[0]:.1f} RY:{step[1]:.1f} RZ:{step[2]:.1f}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 손 크기 정보 출력
                size_info = f"Hand Size: {int(curr_size)} (Target: {int(TARGET_HAND_SIZE)})"
                cv2.putText(frame, size_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # ★★★ 복구된 연결선 (화면 중앙 ~ 손 중심) ★★★
                cv2.line(frame, (320, 240), (cx, cy), (0, 255, 255), 2)
                # ---------------------------------------------------------

        # 메시지 발행
        msg_data = step + [float(mode_id)]
        msg = Float32MultiArray()
        msg.data = msg_data
        self.publisher_.publish(msg)

        cv2.imshow("Webcam Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

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