import os
import cv2
import mediapipe as mp
import numpy as np
import csv

DATA_DIR = r'C:\\Project\\Model\\data'
CSV_PATH = 'exercise_pose_dataset.csv'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['angle_knee', 'angle_hip', 'angle_elbow', 'label'])

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                try:
                    hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    angle_knee = calculate_angle(hip, knee, ankle)
                    angle_hip = calculate_angle(shoulder, hip, knee)
                    angle_elbow = calculate_angle(shoulder, elbow, wrist)

                    writer.writerow([angle_knee, angle_hip, angle_elbow, label])
                    print(f'Saved: {img_name} | {label}')
                except:
                    print(f'Skipped: {img_name} (incomplete landmarks)')
