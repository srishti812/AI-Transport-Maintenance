import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier

# Load Dataset
data = pd.read_csv("transport_data.csv")

print(data.head())

# Features
X = data[[
    'Distance',
    'Temperature',
    'Oil_Level',
    'Vibration'
]]

# Target
y = data['Maintenance']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# LightGBM Model
model = LGBMClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=8,

    num_leaves=31,

    min_child_samples=5,

    random_state=42

)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("✅ Accuracy:", round(accuracy * 100, 2), "%")

# Save Model
joblib.dump(model, "maintenance_model.pkl")

print("✅ LightGBM Model Saved Successfully")