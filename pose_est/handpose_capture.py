import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# 저장된 제스처의 각도 데이터
saved_angles = None

# ==========================================
# [핵심] 3점 사이의 각도 계산 (3D)
# ==========================================
def calculate_angle_3d(a, b, c):
    """
    a, b, c는 각각 랜드마크 객체 (x, y, z)
    b가 중간 지점(관절)
    """
    # 벡터 BA (b -> a)
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    # 벡터 BC (b -> c)
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    # 1. 벡터의 내적 (Dot Product)
    dot_product = np.dot(ba, bc)
    
    # 2. 벡터의 크기 (Norm)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # 3. 코사인 법칙 역함수 (Arccos)
    # 분모가 0이 되는 것을 방지
    if norm_ba * norm_bc == 0:
        return 0.0
        
    cosine_angle = dot_product / (norm_ba * norm_bc)
    
    # 부동소수점 오차로 -1 ~ 1 범위를 벗어나는 경우 처리
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # 라디안 -> 도(Degree) 변환
    return np.degrees(angle)

# ==========================================
# 손가락별 핵심 각도 추출 함수
# ==========================================
def get_hand_angles(landmarks):
    """
    5개 손가락의 굽힘 각도를 리스트로 반환 (총 5개 값)
    완전히 펴지면 180도에 가깝고, 굽히면 0도에 가까움
    """
    angles = []
    
    # 랜드마크 리스트 (편의상 lm으로 참조)
    lm = landmarks.landmark

    # 1. 엄지 (Thumb): CMC(1) - MCP(2) - IP(3) 사이 각도
    # 엄지는 구조가 달라서 관절 선택이 중요함
    angles.append(calculate_angle_3d(lm[1], lm[2], lm[3]))

    # 2. 검지 (Index): MCP(5) - PIP(6) - DIP(7)
    angles.append(calculate_angle_3d(lm[5], lm[6], lm[7]))

    # 3. 중지 (Middle): MCP(9) - PIP(10) - DIP(11)
    angles.append(calculate_angle_3d(lm[9], lm[10], lm[11]))

    # 4. 약지 (Ring): MCP(13) - PIP(14) - DIP(15)
    angles.append(calculate_angle_3d(lm[13], lm[14], lm[15]))

    # 5. 소지 (Pinky): MCP(17) - PIP(18) - DIP(19)
    angles.append(calculate_angle_3d(lm[17], lm[18], lm[19]))

    return np.array(angles)

# ==========================================

print("=== 3D 각도 기반 제스처 인식 ===")
print("k: 현재 손 모양 저장")
print("r: 리셋")
print("ESC: 종료")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # 좌우 반전 및 RGB 변환
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    curr_angles = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # [1] 현재 손의 각도 추출 (5개의 숫자)
            curr_angles = get_hand_angles(hand_landmarks)

            # 디버깅: 각도 값 화면 표시
            # cv2.putText(image, str(np.round(curr_angles, 0)), (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # [2] 저장된 각도와 비교
            if saved_angles is not None:
                # 각도 차이의 평균을 구함 (Mean Absolute Error)
                # 예: 검지가 10도 차이나고 약지가 5도 차이나면 평균 7.5도 차이
                diff = np.mean(np.abs(curr_angles - saved_angles))
                
                cv2.putText(image, f"Angle Diff: {diff:.1f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 임계값 설정 (예: 평균 15도 이내의 차이면 같은 동작)
                if diff < 15.0:
                    cv2.putText(image, "MATCH!", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    # == 매니퓰레이터 동작 ==
                else:
                    cv2.putText(image, "...", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if saved_angles is not None:
        cv2.putText(image, "Pose Saved", (480, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('3D Angle Gesture', image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
    elif key == ord('k'):
        if curr_angles is not None:
            saved_angles = curr_angles
            print(f"저장된 각도: {np.round(saved_angles, 1)}")
    elif key == ord('r'):
        saved_angles = None
        print("리셋됨")

cap.release()
cv2.destroyAllWindows()