import pandas as pd
from sklearn.datasets import load_iris


# Dataset Load (Iris from sklearn)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())
print(df.describe())
print(df['species'].value_counts())

from sklearn.preprocessing import StandardScaler

X = df[iris.feature_names]   # features
y = df['target']              # labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.show()

import joblib

# Save
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')

# Load and predict on new data
loaded_model = joblib.load('iris_model.pkl')
loaded_scaler = joblib.load('iris_scaler.pkl')

new_flower = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
new_scaled = loaded_scaler.transform(new_flower)
prediction = loaded_model.predict(new_scaled)
print(f"Predicted species: {iris.target_names[prediction[0]]}")





