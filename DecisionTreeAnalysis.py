import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns

#Load data
dfa = pd.read_excel('FlipMart Sales DATA set.xlsx')
df = dfa.dropna()

#Define features and target variable
X = df[["Sales"]]
Y = df[["Profit"]]

#Standardization
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)

#Normalization
scaler_normal = MinMaxScaler()
X_normalized = scaler_normal.fit_transform(X_standardized)

#Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.3, random_state=40)

#Train Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=5, random_state=40)  # Limiting depth to avoid overfitting
dt_model.fit(X_train, Y_train)

#Predict
Y_pred = dt_model.predict(X_test)

#Calculate R² score
r2 = r2_score(Y_test, Y_pred)
acc = r2 * 100
print(f"Decision Tree Accuracy: {acc:.2f}%")

# Plot Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Decision Tree Regression - Actual vs Predicted")
plt.show()
#Decision Tree Accuracy: 14.54%