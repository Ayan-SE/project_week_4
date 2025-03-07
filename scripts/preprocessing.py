import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer

def impute_categorical_missingValue_with_mode(df, columns):
      """
      Imputes missing values in categorical columns with their respective modes.
      """
      df = df.copy()  # Avoid modifying the original DataFrame
      for col in columns:
        mode_value = df[col].mode()[0]  # Find the mode of the column
        df[col] = df[col].fillna(mode_value) 
      return df

def impute_categorical_missing_with_placeholder(df, columns,placeholder='Unknown'):
      """
       Imputes missing values in categorical columns with a specified placeholder.
      """
      df = df.copy()  # Avoid modifying the original DataFrame
      for col in columns:
        df[col] = df[col].fillna(placeholder)
      return df

def plot_histograms(data, num_cols, bins=10, figsize=(4, 2)):
     """
     Plots histograms for specified numerical columns in a DataFrame.
    """
     for col in num_cols:
       plt.figure(figsize=figsize)
       plt.hist(data[col], bins=bins)
       plt.xlabel(col)
       plt.ylabel('Frequency')
       plt.title(f'Histogram of {col}')
       plt.show()