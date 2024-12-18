import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the tight_layout warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

# Load and preprocess the dataset
file_path = r"C:\Users\Melat\Documents\ML first assignment\Agricultural_Ethiopia_Indicators.csv"
data = pd.read_csv(file_path, header=1, index_col=0)

# Clean column names
data.columns = data.columns.str.strip()

# Convert all columns to numeric, replacing non-numeric values with NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Check if 'Agricultural exports, mln. US$' column exists before interpolation
if "Agricultural exports, mln. US$" in data.columns:
    data["Agricultural exports, mln. US$"] = data["Agricultural exports, mln. US$"].interpolate(method='linear')
else:
    print("Column 'Agricultural exports, mln. US$' not found in the dataset.")

# Transpose data for easier feature analysis
data_transposed = data.T
print(data_transposed.head())
# Ensure the index is strings for proper plotting
data_transposed.index = data_transposed.index.astype(str)

# Select key features for visualization
features_to_compare = [
     "Total Population (millions)",
    "Rural population (% of total population)",
    "Rural population growth (annual %)",
    "Agriculture, value added (% of GDP)",
    "Agricultural land (% of land area)",
    "Fertilizer consumption (kg/ha of arable land)",
    "Roads, goods transported (million ton-km)",
    "Irrigated Land Area (% of Arable Land)",
    "Agricultural exports, mln. US$",
]

# Filter dataset to include only the selected features (if they exist)
existing_features = [feature for feature in features_to_compare if feature in data_transposed.columns]
selected_data = data_transposed[existing_features]

# Correlation analysis
correlation_matrix = selected_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Selected Features")
plt.show()

# Pairwise scatter plot with trend lines (regplot)
sns.pairplot(selected_data)
plt.suptitle("Pairwise Relationships Between Key Indicators", y=1.02)
plt.show()

# Plot specific relationships with regression lines
for feature in existing_features:
    if feature != "Agricultural exports, mln. US$":
        plt.figure(figsize=(10, 6))
        # Check if 'Agricultural exports, mln. US$' exists to avoid issues
        if "Agricultural exports, mln. US$" in selected_data.columns:
            sns.regplot(
                x=selected_data[feature], 
                y=selected_data["Agricultural exports, mln. US$"], 
                scatter_kws={'alpha':0.7}, 
                line_kws={'color': 'red'},
                ci=None
            )
            plt.title(f'Relationship Between {feature} and Agricultural Exports')
            plt.xlabel(feature)
            plt.ylabel("Agricultural exports, mln. US$")
            plt.grid()
            plt.show()
        else:
            print(f"Skipping plot for {feature} vs Agricultural exports, mln. US$ as the latter is missing.")
