import cv2
import mediapipe as mp
import numpy as np
import math

# 1. MediaPipe 초기화 [cite: 8]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# 제스처 ID 정의 (논문의 Figure 10 참고 [cite: 246, 403])
# 1: 펴짐(Open), 0: 굽힘(Close)
# 순서: [엄지, 검지, 중지, 약지, 소지]
GESTURE_MAP = {
    (1, 1, 1, 1, 1): "Open Menu (ID: 0)",   # [cite: 404]
    (0, 0, 0, 0, 0): "Close Menu (ID: 1)",
    (0, 1, 0, 0, 0): "Menu 1 (ID: 2)",
    (0, 1, 1, 0, 0): "Menu 2 (ID: 3)",
    (0, 1, 1, 1, 0): "Menu 3 (ID: 4)",
    (1, 1, 1, 1, 0): "Menu 4 (ID: 5)", # 엄지 포함 4개
    (1, 1, 0, 0, 1): "Spiderman (ID: 6)", # 
    (0, 1, 0, 0, 1): "Menu 7 (ID: 7)",
    (1, 1, 0, 0, 0): "Menu 8 (ID: 8)",
}

# 두 점 사이의 유클리드 거리 계산 함수
def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# =========================================================
# [핵심 로직] 논문의 Rule-based 방식을 회전 불변형으로 개선
# =========================================================
def get_finger_status(landmarks):
    """
    각 손가락이 펴졌는지(1), 굽혀졌는지(0) 리스트로 반환
    논문의 Figure 6 Pseudocode 를 개선하여 구현
    """
    fingers = []
    lm = landmarks.landmark
    
    # 손가락 끝(Tip) 인덱스: [4, 8, 12, 16, 20] [cite: 186]
    # 비교할 관절(PIP/MCP) 인덱스: Tip - 2  
    # (엄지는 구조가 달라 Tip - 1과 비교하거나 별도 처리 필요하지만, 
    # 여기서는 논문의 'Tip vs Tip-2' 로직을 거리 기반으로 변환하여 적용)
    
    tips = [4, 8, 12, 16, 20]
    
    for i, tip_idx in enumerate(tips):
        # 1. 엄지 (Thumb) - 4번
        if i == 0:
            # 엄지는 '손목(0)' 기준으로 굽힘 여부 판단이 어려우므로
            # 새끼손가락 뿌리(17)와의 거리를 비교하거나, 
            # 관절(IP, 3번)과 끝점(4번)의 벡터 각도를 보는 것이 정확하나,
            # 논문의 단순화된 규칙을 따르되 거리 비교로 대체합니다.
            
            # 엄지 끝(4)이 새끼손가락 뿌리(17)보다 검지 뿌리(5)에 가까우면 펴진 것?
            # 회전에 강한 방식: 엄지 끝(4)이 검지 관절(2)보다 멀리 있는가?
            dist_tip = get_distance(lm[4], lm[17])
            dist_ip = get_distance(lm[3], lm[17])
            
            # 엄지가 펴지면 새끼손가락에서 멀어짐
            if dist_tip > dist_ip:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # 2. 나머지 손가락 (검지~소지)
        else:
            # 논문: hand[tip].y < hand[tip-2].y (화면 위쪽이면 Open) 
            # 개선: 손목(0)에서 끝점(Tip)까지의 거리가, 손목에서 중간관절(PIP)까지보다 먼가?
            
            pip_idx = tip_idx - 2 # 
            
            dist_wrist_to_tip = get_distance(lm[0], lm[tip_idx])
            dist_wrist_to_pip = get_distance(lm[0], lm[pip_idx])
            
            # 손가락을 펴면 손목에서 끝까지 거리가 길어짐
            if dist_wrist_to_tip > dist_wrist_to_pip:
                fingers.append(1) # Open
            else:
                fingers.append(0) # Close

    return tuple(fingers) # 튜플로 반환 (딕셔너리 키로 쓰기 위해)

# =========================================================

print("=== 논문 기반 제스처 인식 (회전 보정됨) ===")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # 좌우 반전 및 RGB 변환
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. 손가락 상태 추출 (0 또는 1) 
            finger_state = get_finger_status(hand_landmarks)
            
            # 2. 매핑된 제스처 이름 찾기 [cite: 65]
            gesture_name = GESTURE_MAP.get(finger_state, "Unknown")
            
            # 화면 그리기
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 텍스트 출력
            state_str = str(finger_state) # 예: (1, 0, 0, 0, 0)
            
            # 상태 표시 (0, 1)
            cv2.putText(image, f"State: {state_str}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # 인식된 제스처 표시
            if gesture_name != "Unknown":
                cv2.putText(image, f"CMD: {gesture_name}", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(image, "Unknown Pose", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Paper Based Gesture', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()