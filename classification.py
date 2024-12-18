import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load and preprocess the dataset
file_path = r"C:\Users\Melat\Documents\ML first assignment\Agricultural_Ethiopia_Indicators.csv"
data = pd.read_csv(file_path, header=1, index_col=0)

# Clean column names
data.columns = data.columns.str.strip()
print(data.head())
print(data.columns)

# Transpose data for easier feature analysis
data_transposed = data.T
# Print out the transposed data to check the structure
print(data_transposed.head())
# Ensure the index is strings for proper plotting
data_transposed.index = data_transposed.index.astype(str)

# Convert all columns to numeric, replacing non-numeric values with NaN
data_transposed = data_transposed.apply(pd.to_numeric, errors='coerce')

# Interpolate missing values for time-series data
data_transposed.interpolate(method='linear', inplace=True)

# Check the cleaned data
print(data_transposed.head())
print(data_transposed.columns)

# Feature Engineering (e.g., calculating Population Growth Rate, Arable Land per Capita, etc.)
data_transposed['Population Growth Rate'] = data_transposed['Total Population (millions)'].pct_change().fillna(0)
data_transposed['Agricultural Yield per Capita'] = data_transposed['Agricultural exports, mln. US$'] / data_transposed['Total Population (millions)']
data_transposed['Arable Land per Capita'] = (data_transposed['Arable land (% of land area)'] / 100) * data_transposed['Land area (1000 sq. km)'] / data_transposed['Total Population (millions)']
# Feature Engineering: Calculate Agricultural Yield per Capita based on Agriculture Value Added (% of GDP)
data_transposed['Agricultural Yield per Capita'] = data_transposed['Agriculture, value added (% of GDP)'] / data_transposed['Total Population (millions)']

# Calculate Growth Rates
data_transposed['Agricultural Yield Growth Rate'] = data_transposed['Agricultural Yield per Capita'].pct_change().fillna(0)
data_transposed['Population Growth Rate'] = data_transposed['Total Population (millions)'].pct_change().fillna(0)

# Define Agricultural Growth Sustainability: If Agricultural Yield Growth Rate >= Population Growth Rate
data_transposed['Sustainability'] = np.where(data_transposed['Agricultural Yield Growth Rate'] >= data_transposed['Population Growth Rate'], 1, 0)

# Check the new classification
print(data_transposed['Sustainability'].value_counts())

# Define sustainability: Net agricultural trade > 0 is sustainable
# Define a stricter threshold for sustainability
#data_transposed['Sustainability'] = np.where(data_transposed['Net agricultural trade, mln. US$'] > 100, 1, 0)

print(data_transposed['Sustainability'].value_counts())
print(data_transposed['Net agricultural trade, mln. US$'])

print(data_transposed['Sustainability'].unique())
print(data_transposed['Sustainability'].value_counts())


# Adding additional datasets for "Main Crops (2013) - Production" and "Main Crops (2011) - Domestic Supply"
# Assuming these datasets are separate CSV files or part of the existing dataset
# Ensure "Production (1000 tonnes)" is a row (index) in the transposed DataFrame
if 'Production (1000 tonnes)' in data_transposed.index:
    production = data_transposed.loc['Production (1000 tonnes)']
    print(production)
else:
    print("'Production (1000 tonnes)' not found in the index.")


# Check the exact column names to match with the features list
print(data_transposed.index)

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

# Check which features are available in the dataset
available_features = [feature for feature in features if feature in data_transposed.columns]
print(f"Available features for model: {available_features}")

# If there are no available features, you'll need to adjust the feature list
if not available_features:
    print("No valid features available in the dataset. Please check the column names.")
else:
    print(data_transposed[available_features].columns)
    print(data_transposed[available_features].shape)  # Ensure there are rows and columns in the selected features
   

    # Proceed with model training if valid features exist
    X = data_transposed[available_features]
    y = data_transposed['Sustainability']  # Assuming sustainability is available
    print(X.shape)
    print(y.shape)
    print(y.unique())

    # Train-test split
    if X.empty or y.empty:
        print("Feature set or target variable is empty!")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training: Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    # Check the feature importances
    importances = clf.feature_importances_
    print(importances)
    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluation
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    # Feature Importance Visualization
    feature_importances = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar', color='skyblue')
    plt.title("Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
