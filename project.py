import pandas as pd

df = pd.read_csv('/Users/patriciagoetz/Desktop/MASTER/SPRING/ML/movies.csv')

# --- Merge with IMDb dataset (before cleaning, to maximize matches) ---
imdb = pd.read_csv('/Users/patriciagoetz/Desktop/MASTER/SPRING/ML/IMDb_All_Genres_etf_clean1.csv')

# Normalize titles for case-insensitive matching
df['_title_key'] = df['title'].str.lower().str.strip()
imdb['_title_key'] = imdb['Movie_Title'].str.lower().str.strip()

# Extract year for disambiguation
df['_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
imdb['_year'] = pd.to_numeric(imdb['Year'], errors='coerce')

# Deduplicate IMDb on title+year: keep highest-rated entry per title+year
imdb = imdb.sort_values('Rating', ascending=False).drop_duplicates(subset=['_title_key', '_year'], keep='first')

# Join on title + year to avoid false matches from shared generic titles
imdb_cols = ['_title_key', '_year', 'Rating', 'Censor', 'Director', 'Total_Gross']
df = df.merge(imdb[imdb_cols], on=['_title_key', '_year'], how='left')
df.drop(columns=['_title_key', '_year'], inplace=True)

df.to_csv('/Users/patriciagoetz/Desktop/MASTER/SPRING/ML/movies_merged.csv', index=False)
print("Saved merged dataset to movies_merged.csv")

# Keep only rows with an observed IMDb match
df = df[df['Rating'].notna()].copy()

print(f"Rows after merge: {len(df)}")
print(f"IMDb Rating matched: {df['Rating'].notna().sum()} / {len(df)}")

# Keep only rows where the key ML targets/features actually exist
df_clean = df[
    (df['budget'] > 0) &
    (df['revenue'] > 0) &
    (df['runtime'] > 0)
].copy()

# Add your target variable: ROI
df_clean['roi'] = df_clean['revenue'] / df_clean['budget']
df_clean['profitable'] = (df_clean['roi'] > 1).astype(int)  # 1 = profitable

print(f"Usable rows after filtering: {len(df_clean)}")
print(df_clean[['title', 'budget', 'revenue', 'roi', 'profitable']].head(10))

df_clean.to_csv('/Users/patriciagoetz/Desktop/MASTER/SPRING/ML/movies_clean.csv', index=False)

print(df_clean['profitable'].value_counts())
print(df_clean['profitable'].value_counts(normalize=True).round(3))
print(df_clean['roi'].describe())
print(f"\nExtreme outliers (ROI > 100): {(df_clean['roi'] > 100).sum()}")
print(f"Suspicious budget entries (budget < 10k): {(df_clean['budget'] < 10000).sum()}")

df_model = df_clean[
    (df_clean['budget'] >= 10000) &  # removes fake $1 budget entries
    (df_clean['roi'] <= 500)          # removes extreme outliers (keeps 99.9% of data)
].copy()

print(f"Rows after cleaning: {len(df_model)}")
print(f"\nROI distribution after cleaning:")
print(df_model['roi'].describe())

print(f"\nClass balance after cleaning:")
print(df_model['profitable'].value_counts())
print(df_model['profitable'].value_counts(normalize=True).round(3))