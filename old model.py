import cv2
import mediapipe as mp
import numpy as np
import math
import joblib

model = joblib.load('pose_model.pkl')

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ฟังก์ชันหามุมระหว่างจุด 3 จุด
def calculate_angle(a, b, c):
    a = np.array(a)  # ข้อสะโพก
    b = np.array(b)  # หัวเข่า
    c = np.array(c)  # ข้อเท้า

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# ตัวแปรนับ squat
counter = 0
stage = None

# เปิดกล้อง
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # แปลงสี
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # ตรวจจับ pose
        results = pose.process(image)

        # วาด keypoints
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # เอาตำแหน่งสะโพก เข่า ข้อเท้า (ด้านขวา)
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # คำนวณมุมเข่า
            angle = calculate_angle(hip, knee, ankle)

            # แสดงมุมที่หน้าจอ
            cv2.putText(image, str(round(angle, 2)),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            features = np.array([[angle_knee, angle_hip, angle_elbow , label]])
            predicted_pose = model.predict(features)[0]

            cv2.putText(image, f'Pose: {predicted_pose}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except:
            pass

        # แสดงข้อมูล
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        cv2.putText(image, 'REPS', (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # วาด skeleton
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # แสดงภาพ
        cv2.imshow('Squat Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
