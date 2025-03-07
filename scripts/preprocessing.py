import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def impute_categorical_missingValue_with_mode(df, columns):
      """
      Imputes missing values in categorical columns with their respective modes.
      """
      df = df.copy()  # Avoid modifying the original DataFrame
      for col in columns:
       mode_value = columns.mode().iloc[0]  # Use .iloc[0] to access the first element
       df[columns.name].fillna(mode_value, inplace=True)
      return df

def impute_categorical_missing_with_placeholder(df, columns,placeholder='Unknown'):
      """
       Imputes missing values in categorical columns with a specified placeholder.
      """
      df = df.copy()  # Avoid modifying the original DataFrame
      for col in columns:
        df[col] = df[col].fillna(placeholder)
      return df

def detect_and_visualize_outliers(data, column, iqr_threshold=1.5, zscore_threshold=3):
    """
    Detects and visualizes outliers using both IQR and Z-score methods.
    """
    if column not in data.columns:
        print(f"Error: Column '{column}' not found in DataFrame.")
        return

    # IQR Method
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound_iqr = q1 - (iqr_threshold * iqr)
    upper_bound_iqr = q3 + (iqr_threshold * iqr)
    outliers_iqr = (data[column] < lower_bound_iqr) | (data[column] > upper_bound_iqr)

    # Z-score Method
    mean = data[column].mean()
    std = data[column].std()
    z_scores = np.abs((data[column] - mean) / std)
    outliers_zscore = z_scores > zscore_threshold

    # Print Outliers
    print(f"Outliers (IQR, threshold={iqr_threshold}):")
    print(data[outliers_iqr])
    print(f"\nOutliers (Z-score, threshold={zscore_threshold}):")
    print(data[outliers_zscore])

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=data[column])
    plt.title(f'Box Plot of {column}')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=data.index, y=data[column])
    plt.title(f'Scatter Plot of {column}')

    plt.show()
