import cv2
import mediapipe as mp

# 1. 설정 (Setup)
# MediaPipe Hands 솔루션 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,             # 최대 감지할 손의 개수 (ml5 기본값과 유사하게 설정)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

# 창 크기 설정 (640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("종료하려면 'ESC' 키를 누르세요.")

# 2. 그리기 루프 (Draw Loop)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break
    
    image = cv2.flip(image, 1)
    # 성능을 위해 쓰기 불가능으로 설정 및 RGB 변환
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 손 감지 수행 (handPose.detectStart와 유사)
    results = hands.process(image_rgb)

    # 다시 그리기 모드로 변경
    image.flags.writeable = True
    
    # 3. 결과 그리기 (Draw tracked hand points)
    if results.multi_hand_landmarks:
        # 감지된 모든 손을 순회 (for let i = 0; i < hands.length; i++)
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 각 손의 모든 관절(keypoints) 순회 (for let j = 0; ...)
            for landmark in hand_landmarks.landmark:
                # 정규화된 좌표(0.0~1.0)를 픽셀 좌표로 변환
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 초록색 원 그리기
                # p5.js의 circle(x, y, 10)은 지름이 10이므로
                # OpenCV circle(..., 반지름, ...)은 5로 설정
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    # 결과 화면 출력
    cv2.imshow('MediaPipe Hand Pose', image)

    # ESC 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()