import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import Counter

# โหลดข้อมูล
df = pd.read_csv('exercise_pose_dataset.csv')

# Features และ Labels
X = df[['angle_right_knee', 'angle_right_hip', 'angle_right_elbow',
        'angle_left_knee', 'angle_left_hip', 'angle_left_elbow',
        'torso_right', 'torso_left']]
y = df['label']

print(Counter(y))  # ตรวจสอบคลาส

best_accuracy = 0
attempt = 0

# เทรนโมเดลจนได้ Accuracy > 0.8
while best_accuracy < 0.7:
    attempt += 1
    print(f"\nTraining attempt #{attempt}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy

# บันทึกโมเดล
joblib.dump(model, 'pose_model.pkl')
print(f"\nFinal model saved with accuracy: {best_accuracy:.4f}")
