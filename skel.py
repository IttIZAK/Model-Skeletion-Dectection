import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load('pose_model.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

pose_count = {"squat": 0, "situp": 0, "pushup": 0}
current_target_pose = "squat"
last_pose = None
pose_keys = {'1': 'squat', '2': 'situp', '3': 'pushup', '4': 'plank', '5': 'walking lunges'
             , '6': 'mountain climbers', '7': 'side plank', '8': 'Laying leg raises'
             , '9': 'Hollow hold', '10': 'dead bug', '11': 'crunches', '12': 'Bicycle crunch'
             , '13': 'Bird dog', '14': 'Russian twist'
             }

cap = cv2.VideoCapture(0)
cv2.namedWindow("Pose Counter", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            lm = results.pose_landmarks.landmark

            # ดึง keypoints ทั้งซ้ายและขวา
            r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # มุมแขน-ขา ซ้ายขวา
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            features = np.array([[r_knee_angle, r_hip_angle, r_elbow_angle, l_knee_angle, l_hip_angle, l_elbow_angle]])
            predicted_pose = model.predict(features)[0]

            if predicted_pose == current_target_pose and last_pose != predicted_pose:
                pose_count[current_target_pose] += 1
            last_pose = predicted_pose

            cv2.putText(image, f'Current Pose: {predicted_pose}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Target: {current_target_pose.capitalize()}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f'Count: {pose_count[current_target_pose]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception as e:
            print("Error in pose detection:", e)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                connection_drawing_spec={
                    mp_pose.POSE_CONNECTIONS: mp_drawing.DrawingSpec(color=(255,255,255), thickness=1)
                }
            )

            # วาดเส้นแยกด้วยสี
            def draw_line(pt1, pt2, color):
                h, w, _ = image.shape
                x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
                cv2.line(image, (x1, y1), (x2, y2), color, 4)

            draw_line(r_shoulder, r_elbow, (0, 255, 255))  # ไหล่-สอก เหลือง
            draw_line(r_elbow, r_wrist, (0, 204, 204)) #สอก-ข้อมือ เหลืองเข้ม
            draw_line(r_hip, r_knee, (255, 0, 127)) #สะโพก-เข่า ม่วง
            draw_line(r_knee, r_ankle, (153, 0, 76)) #เข่า-ข้อเท้า ม่วงเข้ม

            draw_line(l_shoulder, l_elbow, (0, 204, 0))  # ไหล่-สอก เขียว
            draw_line(l_elbow, l_wrist, (0, 153, 0)) #สอก-ข้อมือ เขียวเข้ม
            draw_line(l_hip, l_knee, (255, 0, 0)) #สะโพก-เข่า น้ำเงิน
            draw_line(l_knee, l_ankle, (153, 0, 0)) #เข่า-ข้อเท้า น้ำเงินเข้ม

        cv2.imshow('Pose Counter', image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in pose_keys:
            current_target_pose = pose_keys[chr(key)]
            print(f"เปลี่ยนท่าที่ต้องการนับเป็น: {current_target_pose}")

cap.release()
cv2.destroyAllWindows()
