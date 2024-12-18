import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load and preprocess the dataset
file_path = r"C:\Users\Melat\Documents\ML first assignment\Agricultural_Ethiopia_Indicators.csv"
data = pd.read_csv(file_path, header=1, index_col=0)

# Clean column names
data.columns = data.columns.str.strip()
st.write("Dataset Head", data.head())

# Transpose data for easier feature analysis
data_transposed = data.T
data_transposed.index = data_transposed.index.astype(str)
data_transposed = data_transposed.apply(pd.to_numeric, errors='coerce')
data_transposed.interpolate(method='linear', inplace=True)

# Feature Engineering
data_transposed['Population Growth Rate'] = data_transposed['Total Population (millions)'].pct_change().fillna(0)
data_transposed['Agricultural Yield per Capita'] = data_transposed['Agricultural exports, mln. US$'] / data_transposed['Total Population (millions)']
data_transposed['Arable Land per Capita'] = (data_transposed['Arable land (% of land area)'] / 100) * data_transposed['Land area (1000 sq. km)'] / data_transposed['Total Population (millions)']
data_transposed['Agricultural Yield per Capita'] = data_transposed['Agriculture, value added (% of GDP)'] / data_transposed['Total Population (millions)']
data_transposed['Agricultural Yield Growth Rate'] = data_transposed['Agricultural Yield per Capita'].pct_change().fillna(0)
data_transposed['Population Growth Rate'] = data_transposed['Total Population (millions)'].pct_change().fillna(0)
data_transposed['Sustainability'] = np.where(data_transposed['Agricultural Yield Growth Rate'] >= data_transposed['Population Growth Rate'], 1, 0)

# Feature Selection
features = [
    'Total Population (millions)',
    'Rural population (% of total population)',
    'Rural population growth (annual %)',
    'Agricultural land (% of land area)',
    'Arable Land per Capita',
    'Fertilizer consumption (kg/ha of arable land)',
    'Agricultural Yield per Capita',
    'Population Growth Rate',
]
available_features = [feature for feature in features if feature in data_transposed.columns]
X = data_transposed[available_features]
y = data_transposed['Sustainability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
st.write("Confusion Matrix", confusion_matrix(y_test, y_pred))
st.write("Classification Report", classification_report(y_test, y_pred))
st.write("Accuracy", accuracy_score(y_test, y_pred))

# Feature Importance Visualization
importances = clf.feature_importances_
feature_importances = pd.Series(importances, index=available_features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title("Feature Importances")
plt.ylabel("Importance")
st.pyplot(plt)
