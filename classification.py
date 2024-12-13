import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
data = pd.read_csv('Agricultural_Ethiopia_Indicators.csv', header=1, index_col=0)

# Clean and Preprocess Data
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
data = data.transpose()

# Define Target Variable
population_growth = data['Rural population growth (annual %)']  # Example feature
yield_increase = population_growth * 10  # Simulated yield increase

# Create categorical target variable based on yield increase
bins = [0, 50, 100, np.inf]  # Define bins for categorization
labels = ['Low', 'Medium', 'High']  # Define labels
data['Yield Increase Category'] = pd.cut(yield_increase, bins=bins, labels=labels)

# Select Features and Target
X = data.drop(columns=['Yield Increase Category', 'Agricultural exports, mln. US$'])
y = data['Yield Increase Category']

# Convert column names to strings
X.columns = X.columns.astype(str)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Classification Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)

# Print Classification Report
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
