# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset using Pandas.

2.Select employee attributes as input features and salary as the target variable.

3.Split the dataset into training and testing sets.

4.Train a Decision Tree Regressor using the training data.

5.Predict salary and evaluate the model using Mean Squared Error and R² score.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Udhaya.s
RegisterNumber:  212225230287
*/
```
~~~ python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Employee dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\Employee.csv")

# Select features and target
X = df[["satisfaction_level", "last_evaluation",
        "number_project", "average_montly_hours",
        "time_spend_company"]]

y = df["salary"]

# Convert salary into numerical values if required
# Example: low = 1, medium = 2, high = 3
if df["salary"].dtype == "object":
    y = df["salary"].map({"low": 1, "medium": 2, "high": 3})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict salary
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
~~~

## Output:
<img width="708" height="64" alt="image" src="https://github.com/user-attachments/assets/e844034e-0858-4563-9bd9-fe7c26e42c86" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
