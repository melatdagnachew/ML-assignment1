import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Step 1: Load Data
data = pd.read_csv('Agricultural_Ethiopia_Indicators.csv' ,header=1,index_col=0)

# Display the first few rows of the dataset to understand its structure
print("Original Data:")
print(data.head())

# Step 2: Clean and Preprocess Data
# Convert all values to numeric, replacing non-numeric with NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Interpolate missing values (using linear interpolation)
data = data.interpolate(method='linear', axis=0)

# Transpose the data for easier feature selection later
data = data.transpose()

# Convert column names to strings
data.columns = data.columns.astype(str)

# Display the cleaned data
print("\nCleaned and Transposed Data:")
print(data.head())

# Step 3: Select Features and Target Variable
target_variable = 'Agricultural exports, mln. US$'
if target_variable not in data.columns:
    raise ValueError(f"{target_variable} not found in the dataset.")

X = data.drop(columns=[target_variable])
y = data[target_variable]

# Step 4: Handle Missing Values in Features
# As we've already interpolated, no need for further missing value handling in X
# Impute target variable if still any NaN values
y.fillna(y.mean(), inplace=True)

# Display the processed features and target variable
print("\nProcessed Features (X):")
print(X.head())
print("\nProcessed Target Variable (y):")
print(y.head())

# Step 5: Visualize Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData has been split into training and testing subsets:")
print(f"Training features: {X_train.shape}, Training target: {y_train.shape}")
print(f"Testing features: {X_test.shape}, Testing target: {y_test.shape}")

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display coefficients for feature importance
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:")
print(coefficients)

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importance from Linear Regression')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Step 9: Plot Predicted vs Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.tight_layout()
plt.show()
