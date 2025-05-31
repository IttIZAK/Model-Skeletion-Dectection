import cv2
import mediapipe as mp
import numpy as np
import joblib

# โหลดโมเดล
model = joblib.load('pose_model.pkl')

# ตั้งค่า Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ฟังก์ชันคำนวณมุม
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ตัวแปรนับท่า
last_pose = None
pose_count = {"squat": 0, "situp": 0, "pushup": 0}
current_target_pose = "squat"

# Key mapping สำหรับเปลี่ยนท่า
pose_keys = {'1': 'squat', '2': 'situp', '3': 'pushup'}

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cv2.namedWindow("Pose Counter", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # แปลงภาพ
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # ดึง keypoints
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # คำนวณมุม
            angle_knee = calculate_angle(hip, knee, ankle)
            angle_hip = calculate_angle(shoulder, hip, knee)
            angle_elbow = calculate_angle(shoulder, elbow, wrist)

            # ทำนายท่า
            features = np.array([[angle_knee, angle_hip, angle_elbow]])
            predicted_pose = model.predict(features)[0]

            # นับเฉพาะท่าที่เลือกไว้
            if predicted_pose == current_target_pose and last_pose != predicted_pose:
                pose_count[current_target_pose] += 1
            last_pose = predicted_pose

            # แสดงข้อมูล
            cv2.putText(image, f'Current Pose: {predicted_pose}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Target: {current_target_pose.capitalize()}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f'Count: {pose_count[current_target_pose]}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception as e:
            print("Error in pose detection:", e)

        # วาด skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # แสดงภาพ
        cv2.imshow('Pose Counter', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in pose_keys:
            current_target_pose = pose_keys[chr(key)]
            print(f"🟢 เปลี่ยนท่าที่ต้องการนับเป็น: {current_target_pose}")

cap.release()
cv2.destroyAllWindows()
