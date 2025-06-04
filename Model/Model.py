import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('exercise_pose_dataset.csv')

X = df[['angle_right_knee', 'angle_right_hip', 'angle_right_elbow',
        'angle_left_knee', 'angle_left_hip', 'angle_left_elbow',
        'torso_right', 'torso_left',
        ]]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, 'pose_model.pkl')
