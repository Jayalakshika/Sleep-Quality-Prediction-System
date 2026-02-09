import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("sleep_data.csv")

# Encode target
le = LabelEncoder()
data["SleepQuality"] = le.fit_transform(data["SleepQuality"])

X = data.drop("SleepQuality", axis=1)
y = data["SleepQuality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "sleep_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved successfully")
