# Simple manual Decision Tree logic using if-else conditions

# Step 1: Create manual dataset
data = [
    {'Study_Hours': 2, 'Attendance': 50, 'Result': 'Fail'},
    {'Study_Hours': 4, 'Attendance': 60, 'Result': 'Fail'},
    {'Study_Hours': 6, 'Attendance': 70, 'Result': 'Pass'},
    {'Study_Hours': 8, 'Attendance': 80, 'Result': 'Pass'},
    {'Study_Hours': 10, 'Attendance': 90, 'Result': 'Pass'},
    {'Study_Hours': 1, 'Attendance': 40, 'Result': 'Fail'},
    {'Study_Hours': 3, 'Attendance': 55, 'Result': 'Fail'},
    {'Study_Hours': 5, 'Attendance': 65, 'Result': 'Pass'},
    {'Study_Hours': 7, 'Attendance': 75, 'Result': 'Pass'},
    {'Study_Hours': 9, 'Attendance': 85, 'Result': 'Pass'}
]

# Step 2: Define a simple decision tree rule (manual logic)
# Rule: If study_hours > 5 and attendance > 65 → Pass else Fail

def predict(study_hours, attendance):
    if study_hours > 5 and attendance > 65:
        return "Pass"
    else:
        return "Fail"

# Step 3: Test predictions on the dataset
correct = 0
for row in data:
    predicted = predict(row['Study_Hours'], row['Attendance'])
    actual = row['Result']
    if predicted == actual:
        correct += 1
    print(f"Study_Hours={row['Study_Hours']}, Attendance={row['Attendance']} → Predicted={predicted}, Actual={actual}")

# Step 4: Calculate accuracy
accuracy = (correct / len(data)) * 100
print("\nModel Accuracy: {:.2f}%".format(accuracy))
