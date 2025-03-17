import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings

# Suppress warning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Parameters for dataset creation
n_samples = 500  # Number of samples
n_features = 5  # Number of features
n_targets = 3  # Number of target variables

# Generate random regression data
X, y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=0.1, random_state=42)

# Define column names for the DataFrame
columns = [f"Feature_{i+1}" for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
for i in range(n_targets):
    df[f"Target_{i+1}"] = y[:, i]

# Introduce missing values in one feature
missing_feature = "Feature_1"  # Feture with missing values
missing_rate = 0.05  # Propartion of missing values
n_missing = int(n_samples * missing_rate)  # Number of missing values
missing_indices = np.random.choice(df.index, n_missing, replace=False)  # Indices of missing values
df.loc[missing_indices, missing_feature] = np.nan

# Random imputation for missing values
feature_min, feature_max = df[missing_feature].min(), df[missing_feature].max()
random_imputed_values = np.random.uniform(feature_min, feature_max, n_missing)
df_random_imputed = df.copy()
df_random_imputed.loc[missing_indices, missing_feature] = random_imputed_values

# Separate rows with and without missing values
df_non_missing = df.dropna()
df_missing = df[df[missing_feature].isna()]

# Impute missing values using a regression model
reg = LinearRegression()
X_train = df_non_missing.drop(columns=[missing_feature])  # İndependent variables
y_train = df_non_missing[missing_feature]  # Target variable
reg.fit(X_train, y_train)  # Train the model

X_missing = df_missing.drop(columns=[missing_feature])  # İndependent variables for missing rows
predicted_values = reg.predict(X_missing)  # Predicted values for missing data

# Add small random noise to predictions
noise = np.random.normal(0, 0.1, size=predicted_values.shape)
predicted_values += noise

# Create a DataFrame with regression-imputed values
df_regression_imputed = df.copy()
df_regression_imputed.loc[missing_indices, missing_feature] = predicted_values

# Visualize predictions for target variables
plt.figure(figsize=(15, 5))

for i in range(1, n_targets + 1):
    actual_values = y_train  # Actual values
    predicted_values_for_target = reg.predict(X_train) + np.random.normal(0, 0.1, len(y_train))

    plt.subplot(1, n_targets, i)
    plt.scatter(range(len(actual_values)), actual_values, color='red', label='Actual value', s=10)
    plt.scatter(range(len(predicted_values_for_target)), predicted_values_for_target, color='blue', label='Predicted value', s=10)
    plt.title(f'Target_{i}')
    plt.xlabel('Index')
    plt.ylabel(f'Target_{i}')
    plt.legend()

plt.tight_layout()
plt.show()

# Calculate training error of the regresion model
mse = mean_squared_error(y_train, reg.predict(X_train))
print(f"Mean Squared Error (MSE): {mse}")

# Save DataFrames to CSV files
df.to_csv('original_dataset.csv', index=False)
df_random_imputed.to_csv('random_imputed_dataset.csv', index=False)
df_regression_imputed.to_csv('regression_imputed_dataset.csv', index=False)

print("Datasets saved.")

# Function to train and evaluate a model on a dataset
def train_and_evaluate(df, target_columns):
    features = df.drop(columns=target_columns)  # Independent variables
    targets = df[target_columns]  # Target variables

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)  # Split data

    model = MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=500, random_state=42)  # Define model
    model.fit(X_train, y_train)  # Train model

    y_pred = model.predict(X_test)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate error
    return mse

# Define target colunms
target_columns = [f"Target_{i+1}" for i in range(n_targets)]

# Evaluate models on diffierent datasets
mse_original = train_and_evaluate(df.dropna(), target_columns)
mse_random_imputed = train_and_evaluate(df_random_imputed, target_columns)
mse_regression_imputed = train_and_evaluate(df_regression_imputed, target_columns)

# Print results
print("Mean Squared Errors (MSE):")
print(f"Original Dataset: {mse_original:.4f}")
print(f"Random Imputed Dataset: {mse_random_imputed:.4f}")
print(f"Regression Imputed Dataset: {mse_regression_imputed:.4f}")
