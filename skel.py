import cv2
import mediapipe as mp
import numpy as np
import joblib
import time  # เพิ่มไลบรารี time สำหรับคูลดาวน์

model = joblib.load('pose_model.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

pose_count = {}
reps_count = {}
current_target_pose = "squat"

pose_state = {"last": None, "ready": True}
reps_state = {"direction": None}  # 'down' → 'up' = 1 rep

# เพิ่มตัวแปรเก็บเวลานับครั้งล่าสุด สำหรับคูลดาวน์
last_count_time = 0
cooldown_seconds = 2  # กำหนดคูลดาวน์ 2 วินาที

pose_keys = {
    '1': 'squat', '2': 'situp', '3': 'pushup', '4': 'plank', '5': 'walking lunges',
    '6': 'mountain climbers', '7': 'side plank', '8': 'Laying leg raises',
    '9': 'Hollow hold', '10': 'dead bug', '11': 'crunches', '12': 'Bicycle crunch',
    '13': 'Bird dog', '14': 'Russian twist'
}

cap = cv2.VideoCapture(0)
cv2.namedWindow("Pose Counter", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# กำหนดสีสำหรับแต่ละส่วน (BGR)
color_right_arm = (0, 0, 255)      # แดง
color_left_arm = (255, 0, 0)       # น้ำเงิน
color_right_leg = (0, 255, 255)    # เหลือง
color_left_leg = (0, 255, 0)       # เขียว
color_back = (128, 0, 128)         # ม่วง
color_other = (200, 200, 200)      # เทาอ่อน

spec_right_arm = mp_drawing.DrawingSpec(color=color_right_arm, thickness=3, circle_radius=5)
spec_left_arm = mp_drawing.DrawingSpec(color=color_left_arm, thickness=3, circle_radius=5)
spec_right_leg = mp_drawing.DrawingSpec(color=color_right_leg, thickness=3, circle_radius=5)
spec_left_leg = mp_drawing.DrawingSpec(color=color_left_leg, thickness=3, circle_radius=5)
spec_back = mp_drawing.DrawingSpec(color=color_back, thickness=3, circle_radius=5)
spec_other = mp_drawing.DrawingSpec(color=color_other, thickness=2, circle_radius=3)

right_arm_ids = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.RIGHT_PINKY
]

left_arm_ids = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.LEFT_THUMB,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.LEFT_PINKY
]

right_leg_ids = [
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]

left_leg_ids = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX
]

back_ids = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
]

back_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value)
]

from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_by_ids(image, landmarks, connections, ids, drawing_spec):
    selected_landmarks = landmark_pb2.NormalizedLandmarkList()
    id_values = [i.value for i in ids]
    for i in id_values:
        selected_landmarks.landmark.append(landmarks.landmark[i])
    filtered_connections = set()
    for connection in connections:
        if connection[0] in id_values and connection[1] in id_values:
            new_start = id_values.index(connection[0])
            new_end = id_values.index(connection[1])
            filtered_connections.add((new_start, new_end))

    mp_drawing.draw_landmarks(
        image,
        selected_landmarks,
        filtered_connections,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec
    )


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

            r_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
            r_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            r_ankle = get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            r_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            r_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            r_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)

            l_hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
            l_knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
            l_ankle = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            l_shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            l_elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            l_wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)

            features = np.array([[
                calculate_angle(r_hip, r_knee, r_ankle),
                calculate_angle(r_shoulder, r_hip, r_knee),
                calculate_angle(r_shoulder, r_elbow, r_wrist),
                calculate_angle(l_hip, l_knee, l_ankle),
                calculate_angle(l_shoulder, l_hip, l_knee),
                calculate_angle(l_shoulder, l_elbow, l_wrist),
                calculate_angle(r_elbow, r_shoulder, l_shoulder),
                calculate_angle(l_elbow, l_shoulder, r_shoulder)
            ]])

            predicted_pose = model.predict(features)[0]

            if current_target_pose not in pose_count:
                pose_count[current_target_pose] = 0
            if current_target_pose not in reps_count:
                reps_count[current_target_pose] = 0

            current_time = time.time()

            if predicted_pose == current_target_pose:
                # เช็คคูลดาวน์ก่อนนับ
                if pose_state["ready"] and pose_state["last"] != current_target_pose:
                    # เช็คว่าผ่านคูลดาวน์ 2 วินาทีหรือยัง
                    if current_time - last_count_time >= cooldown_seconds:
                        pose_count[current_target_pose] += 1
                        last_count_time = current_time
                        pose_state["ready"] = False
            else:
                pose_state["ready"] = True

            pose_state["last"] = predicted_pose

            right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

            if current_target_pose == "squat":
                if right_knee_angle < 90:
                    reps_state["direction"] = "down"
                elif right_knee_angle > 160 and reps_state["direction"] == "down":
                    reps_count[current_target_pose] += 1
                    reps_state["direction"] = "up"

            cv2.putText(image, f'Current Pose: {predicted_pose}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Target: {current_target_pose.capitalize()}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f'Count: {pose_count[current_target_pose]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(image, f'Reps: {reps_count[current_target_pose]}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)

        except Exception as e:
            print("Error:", e)

        if results.pose_landmarks:
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, right_arm_ids, spec_right_arm)
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, left_arm_ids, spec_left_arm)
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, right_leg_ids, spec_right_leg)
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, left_leg_ids, spec_left_leg)
            draw_landmarks_by_ids(image, results.pose_landmarks, back_connections, back_ids, spec_back)
            all_ids = set(range(len(results.pose_landmarks.landmark)))
            arm_leg_back_ids = set([i.value for i in right_arm_ids + left_arm_ids + right_leg_ids + left_leg_ids + back_ids])
            other_ids = all_ids - arm_leg_back_ids
            draw_landmarks_by_ids(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, [mp_pose.PoseLandmark(i) for i in other_ids], spec_other)

        cv2.imshow("Pose Counter", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
