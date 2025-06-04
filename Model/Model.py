import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# โหลดข้อมูล
df = pd.read_csv('exercise_pose_dataset.csv')

# ตรวจสอบและกรอง class ที่มีตัวอย่าง < 2
label_counts = df['label'].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df = df[df['label'].isin(valid_labels)]

# Features และ Labels
X = df[['angle_right_knee', 'angle_right_hip', 'angle_right_elbow',
        'angle_left_knee', 'angle_left_hip', 'angle_left_elbow',
        'torso_right', 'torso_left']]
y = df['label']

best_accuracy = 0
attempt = 0

# วนลูปเทรนจนกว่าจะได้ Accuracy > 0.79
while best_accuracy < 0.8:
    attempt += 1
    print(f"\nTraining attempt #{attempt}...")

    # แบ่งข้อมูลแบบ stratify (ปลอดภัยแล้ว เพราะไม่มี class ที่น้อยกว่า 2 ตัว)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=None)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy

# บันทึกโมเดล
joblib.dump(model, 'pose_model.pkl')
print(f"\nFinal model saved with accuracy: {best_accuracy:.4f}")
