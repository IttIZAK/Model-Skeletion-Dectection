import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from mediapipe.framework.formats import landmark_pb2

# โหลดโมเดล
model = joblib.load('pose_model.pkl')

# เริ่มต้น MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# กำหนดสีสำหรับแต่ละส่วน (BGR) ของร่างกาย
color_right_arm = (0, 0, 255)      # แดง
color_left_arm = (255, 0, 0)       # น้ำเงิน
color_right_leg = (0, 255, 255)    # เหลือง
color_left_leg = (0, 255, 0)       # เขียว
color_back = (128, 0, 128)         # ม่วง
color_other = (200, 200, 200)      # เทาอ่อน

# สร้าง DrawingSpec
specs = {
    "right_arm": mp_drawing.DrawingSpec(color=color_right_arm, thickness=3, circle_radius=5),
    "left_arm": mp_drawing.DrawingSpec(color=color_left_arm, thickness=3, circle_radius=5),
    "right_leg": mp_drawing.DrawingSpec(color=color_right_leg, thickness=3, circle_radius=5),
    "left_leg": mp_drawing.DrawingSpec(color=color_left_leg, thickness=3, circle_radius=5),
    "back": mp_drawing.DrawingSpec(color=color_back, thickness=3, circle_radius=5),
    "other": mp_drawing.DrawingSpec(color=color_other, thickness=2, circle_radius=3)
}

# ฟังก์ชันวาดเฉพาะ landmark ตามกลุ่ม
def draw_landmarks_by_ids(image, landmarks, connections, ids, drawing_spec):
    selected = landmark_pb2.NormalizedLandmarkList()
    id_vals = [i.value for i in ids]
    for i in id_vals:
        selected.landmark.append(landmarks.landmark[i])
    filtered_conn = {(id_vals.index(c[0]), id_vals.index(c[1]))
                     for c in connections if c[0] in id_vals and c[1] in id_vals}
    mp_drawing.draw_landmarks(image, selected, filtered_conn, drawing_spec, drawing_spec)

# รายชื่อ landmark ของแต่ละส่วน
right_arm_ids = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
                 mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB,
                 mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.RIGHT_PINKY]
left_arm_ids = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB,
                mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.LEFT_PINKY]
right_leg_ids = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                 mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
left_leg_ids = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
back_ids = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
back_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value)
]

# คำนวณมุม
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# การตั้งค่าเริ่มต้น
pose_count = {}
reps_count = {}
pose_state = {"last": None, "ready": True}
reps_state = {"direction": None}
cooldown_seconds = 2
last_count_time = 0
current_target_pose = "squat"

pose_keys = {
    '1': 'squat', '2': 'situp', '3': 'pushup', '4': 'plank',
    '5': 'walking lunges', '6': 'mountain climbers', '7': 'side plank',
    '8': 'Laying leg raises', '9': 'Hollow hold', '0': 'dead bug',
    'a': 'crunches', 'b': 'Bicycle crunch', 'c': 'Bird dog', 'd': 'Russian twist'
}

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cv2.namedWindow("Pose Counter", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

            def get_coords(idx):
                return [lm[idx].x, lm[idx].y]

            # มุมสำหรับ feature
            features = np.array([[
                calculate_angle(get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.LEFT_HIP.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_HIP.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)),
                calculate_angle(get_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                                get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value))
            ]])

            predicted_pose = model.predict(features)[0]

            if current_target_pose not in pose_count:
                pose_count[current_target_pose] = 0
            if current_target_pose not in reps_count:
                reps_count[current_target_pose] = 0

            now = time.time()
            if predicted_pose == current_target_pose:
                if pose_state["ready"] and pose_state["last"] != current_target_pose and now - last_count_time >= cooldown_seconds:
                    pose_count[current_target_pose] += 1
                    last_count_time = now
                    pose_state["ready"] = False
            else:
                pose_state["ready"] = True
            pose_state["last"] = predicted_pose

            # เพิ่มการนับ rep แบบลงแล้วขึ้น
            right_knee_angle = calculate_angle(get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                               get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                                               get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value))
            if current_target_pose == "squat":
                if right_knee_angle < 90:
                    reps_state["direction"] = "down"
                elif right_knee_angle > 160 and reps_state["direction"] == "down":
                    reps_count[current_target_pose] += 1
                    reps_state["direction"] = "up"

            cv2.putText(image, f'Current Pose: {predicted_pose}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Target: {current_target_pose.capitalize()}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f'Count: {pose_count[current_target_pose]}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(image, f'Reps: {reps_count[current_target_pose]}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)

        except Exception as e:
            print("Error:", e)

        if results.pose_landmarks:
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, right_arm_ids, specs["right_arm"])
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, left_arm_ids, specs["left_arm"])
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, right_leg_ids, specs["right_leg"])
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, left_leg_ids, specs["left_leg"])
            draw_landmarks_by_ids(image, results.pose_landmarks, back_connections, back_ids, specs["back"])

            all_ids = set(range(len(results.pose_landmarks.landmark)))
            exclude_ids = set(i.value for i in right_arm_ids + left_arm_ids + right_leg_ids + left_leg_ids + back_ids)
            other_ids = [mp_pose.PoseLandmark(i) for i in all_ids - exclude_ids]
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, other_ids, specs["other"])

        cv2.imshow("Pose Counter", image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if 32 <= key <= 126 and chr(key) in pose_keys:
            current_target_pose = pose_keys[chr(key)]
            print(f"เปลี่ยนท่าที่ต้องการนับเป็น: {current_target_pose}")

cap.release()
cv2.destroyAllWindows()
