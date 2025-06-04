import os
import cv2
import mediapipe as mp
import numpy as np
import csv

DATA_DIR = r'C:\Project\Model\data'
CSV_PATH = 'exercise_pose_dataset.csv'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'angle_right_knee', 'angle_right_hip', 'angle_right_elbow',
        'angle_left_knee', 'angle_left_hip', 'angle_left_elbow',
        'torso_right', 'torso_left',
        'label'
    ])


    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            # แปลงภาพที่ไม่ใช่ jpg เป็น jpg
            ext = os.path.splitext(img_name)[1].lower()
            if ext not in ['.jpg', '.jpeg']:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                new_img_name = os.path.splitext(img_name)[0] + ".jpg"
                new_img_path = os.path.join(folder, new_img_name)
                cv2.imwrite(new_img_path, img)
                os.remove(img_path)
                img_path = new_img_path
                img_name = new_img_name

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                try:
                    # ขวา
                    r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # ซ้าย
                    l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # แขนขวา-ซ้าย
                    angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)
                    angle_r_hip = calculate_angle(r_shoulder, r_hip, r_knee)
                    angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    torso_right = calculate_angle(r_shoulder, r_hip, r_knee)

                    # ขาขวา-ซ้าย
                    angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
                    angle_l_hip = calculate_angle(l_shoulder, l_hip, l_knee)
                    angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    torso_left = calculate_angle(l_shoulder, l_hip, l_knee)

                    writer.writerow([
                        angle_r_knee, angle_r_hip, angle_r_elbow,
                        angle_l_knee, angle_l_hip, angle_l_elbow,
                        torso_right, torso_left,
                        label
                    ])
                    print(f'Saved: {img_name} | {label}')
                except:
                    print(f'Skipped: {img_name} (incomplete landmarks)')
