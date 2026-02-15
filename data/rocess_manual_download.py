# src/data/process_manual_download.py
import pandas as pd

# Load manually downloaded file
df = pd.read_csv('data/raw/time_series_60min_singleindex.csv', 
                  parse_dates=[0], 
                  index_col=0)

# Filter German data
german_cols = [col for col in df.columns if 'DE_load' in col]
df_germany = df[german_cols]

# Save
df_germany.to_csv('data/raw/germany_energy_consumption.csv')
print(f"✅ Processed {len(df_germany):,} rows")
