# Experiment No: 08
# Aim: Implementation of Naive Bayes Classifier using Python

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Combine for visualization
data = pd.concat([X, y], axis=1)

# 3. Data Preprocessing & Visualization
# Pairplot before scaling
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Split data (train-test split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Iris Features")
plt.show()

# 4. Train Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# 5. Make Predictions
y_pred = nb.predict(X_test_scaled)

# 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Gaussian Naive Bayes: {:.4f}".format(accuracy))

cm = confusion_matrix(y_test, y_pred)

# 7. Visualize Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Conclusion
print("We have successfully implemented the Naive Bayes Classifier using Python.")
