import pandas as pd
import logging

# Logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv_file(file_path):
    """Loads a CSV file, removes unnamed columns, and handles errors."""
    
    logging.info("Loading data from file.......")

    try:
        # Load CSV file
        data = pd.read_csv(file_path)

        # Remove unnamed columns
        data = data.loc[:, ~data.columns.str.contains('^unnamed', case=False)]

        print(f"Dataset loaded successfully from {file_path}")
        logging.info(f"Data Loaded with shape {data.shape}")

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        logging.error("File not found.")
        raise

    except Exception as e:
        print(f"Error loading file: {e}")
        logging.error(f"Error loading file: {e}")
        raise