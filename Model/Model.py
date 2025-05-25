import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# โหลดข้อมูล
df = pd.read_csv('exercise_pose_dataset.csv')

# แยก features และ labels
X = df[['angle_knee', 'angle_hip','angle_elbow']]  # เพิ่ม feature ได้ตามต้องการ
y = df['label']

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# สร้างและฝึกโมเดล
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ทดสอบ
print("Accuracy:", model.score(X_test, y_test))

# บันทึกโมเดล
joblib.dump(model, 'pose_model.pkl')
