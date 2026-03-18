import pandas as pd

# Read the movies CSV file
movies_df = pd.read_csv('/Users/patriciagoetz/Desktop/MASTER/SPRING/ML/movies.csv')

# Display basic info about the dataset
print(movies_df.head())
print(f"\nDataset shape: {movies_df.shape}")
print(f"\nColumn names: {movies_df.columns.tolist()}")
