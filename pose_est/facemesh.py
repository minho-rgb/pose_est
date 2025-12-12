import cv2
import mediapipe as mp

# 1. 설정 (Setup)
# MediaPipe FaceMesh 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,          # options: maxFaces: 1
    refine_landmarks=False,   # options: refineLandmarks: false
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 캡처 시작 (p5.js의 createCapture(VIDEO)와 동일)
cap = cv2.VideoCapture(1)

# 창 크기 설정 (선택 사항, p5.js의 createCanvas와 유사 효과)
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
    # 성능을 위해 이미지를 쓰기 불가능으로 설정하고 RGB로 변환
    # (OpenCV는 BGR을 사용하지만, MediaPipe는 RGB를 입력으로 받음)
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 얼굴 감지 수행 (faceMesh.detectStart와 유사)
    results = face_mesh.process(image_rgb)

    # 이미지를 다시 그리기 모드로 변경 및 BGR로 원복 (화면 출력용)
    image.flags.writeable = True
    
    # 3. 결과 그리기 (Draw tracked face points)
    # faces 배열을 순회하는 것과 동일
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 각 얼굴의 랜드마크(점) 순회
            for landmark in face_landmarks.landmark:
                # 정규화된 좌표(0.0~1.0)를 픽셀 좌표로 변환
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 초록색 점 그리기 (BGR: (0, 255, 0))
                # p5.js의 circle(x, y, 5)와 동일 (반지름 2~3 정도가 적당)
                cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)

    # 결과 화면 출력
    cv2.imshow('MediaPipe Face Mesh', image)

    # ESC 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
