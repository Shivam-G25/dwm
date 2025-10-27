from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Step 1: Create modified manual dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [35, 45, 55, 60, 65, 68, 70, 72, 80, 90],
    'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail']
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Split features and target
X = df[['Study_Hours', 'Attendance']]
y = df['Result']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Step 5: Train the model
model = DecisionTreeClassifier(random_state=2)
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 8: Display results
print("\nTest Data:\n", X_test)
print("\nActual Results:", list(y_test))
print("Predicted Results:", list(y_pred))
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
