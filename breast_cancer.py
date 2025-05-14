import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv('Cancer_Wisconsin.csv')

print("\n=== Data Preview ===")
print(df.head())

print("\n=== Data Description ===")
print(df.describe())

print("\n=== Data Info ===")
print(df.info())

df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

if df.isnull().sum().any():
    print("Warning: Missing values found!")
    print(df.isnull().sum())
    df.dropna(inplace=True)

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importance = pd.Series(
    model.coef_[0], index=X.columns
).sort_values(key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))