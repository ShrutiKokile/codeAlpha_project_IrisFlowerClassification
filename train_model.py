import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
data = pd.read_csv("Iris.csv")

# Remove Id column if present
if "Id" in data.columns:
    data.drop("Id", axis=1, inplace=True)

# Encode Species
encoder = LabelEncoder()
data["Species"] = encoder.fit_transform(data["Species"])

# Features and Target
X = data.drop("Species", axis=1)
y = data["Species"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_scaled, y)

# Save Files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model trained and saved successfully!")

