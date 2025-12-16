import cv2
import mediapipe as mp

# =========================================================
# [설정] 초기화 (JS의 preload & setup 단계에 해당)
# =========================================================

# MediaPipe Pose 모델 로드
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 캡처 시작 (JS의 createCapture(VIDEO))
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(">>> Pose Estimation Started (Press 'q' to exit)")

# =========================================================
# [루프] 프레임 처리 (JS의 draw 단계에 해당)
# =========================================================
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 1. 이미지 처리 준비
    # OpenCV는 BGR을 사용하므로, MediaPipe 처리를 위해 RGB로 변환해야 함
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. 포즈 감지 (JS의 bodyPose.detectStart)
    results = pose.process(image_rgb)
    
    # 이미지 높이, 너비 가져오기 (좌표 계산용)
    h, w, _ = frame.shape

    # 3. 그리기 (포즈가 감지되었을 때만 수행)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # --- [A] 스켈레톤(뼈대) 그리기 (JS의 connections 루프) ---
        # ml5.js 예제의 'stroke(255, 0, 0)' (빨간색 선) 구현
        # mp_pose.POSE_CONNECTIONS에 연결 정보가 들어있음
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # 각 관절의 좌표 가져오기
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            # 신뢰도 체크 (JS의 confidence > 0.1)
            if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                # 정규화된 좌표(0~1)를 픽셀 좌표로 변환
                x1, y1 = int(start_point.x * w), int(start_point.y * h)
                x2, y2 = int(end_point.x * w), int(end_point.y * h)
                
                # 선 그리기 (BGR 색상: (0, 0, 255) -> 빨간색)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --- [B] 키포인트(관절) 그리기 (JS의 pose.keypoints 루프) ---
        # ml5.js 예제의 'fill(0, 255, 0)' (초록색 원) 구현
        for landmark in landmarks:
            # 신뢰도 체크
            if landmark.visibility > 0.5:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 원 그리기 (BGR 색상: (0, 255, 0) -> 초록색)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # 4. 화면 출력 (JS의 image(video, ...))
    cv2.imshow('MediaPipe Pose', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()