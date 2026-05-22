import pandas as pd
import sys
import os

csv_path = os.path.join(os.path.dirname(__file__), "ltpp_processed_data.csv")

print(f"Reading CSV: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print()

# Columns of interest
cols = [
    'MRI', 'IRI_LAG_1', 'IRI_LAG_2', 'PAVEMENT_AGE',
    'LATITUDE', 'LONGITUDE', 'ELEVATION',
    'TOTAL_THICKNESS', 'AC_THICKNESS', 'BASE_THICKNESS', 'NUM_LAYERS',
    'PAVEMENT_FAMILY_ENC',
    'DEGREE_DAYS_OVER_10C_YR', 'COLDEST_AIR_TEMP', 'HIGH_TEMP_7DAYS',
    'MIN_SURFACE_50_TEMP', 'FREEZE_INDEX', 'FREEZE_THAW',
    'PRECIPITATION', 'EVAPORATION'
]

print("=" * 110)
print("DESCRIPTIVE STATISTICS")
print("=" * 110)

# Compute all statistics manually for clarity
for col in cols:
    s = df[col]
    print(f"\n--- {col} ---")
    print(f"  Count:  {s.count():>12,.0f}")
    print(f"  Unique: {s.nunique():>12,.0f}")
    print(f"  Min:    {s.min():>16.8f}")
    print(f"  Max:    {s.max():>16.8f}")
    print(f"  Mean:   {s.mean():>16.8f}")
    print(f"  Median: {s.median():>16.8f}")
    print(f"  Std:    {s.std():>16.8f}")

# Unique SHRP_ID count
shrpid_unique = df['SHRP_ID'].nunique()
print(f"\n{'=' * 110}")
print(f"UNIQUE SHRP_ID count: {shrpid_unique}")
print(f"{'=' * 110}")

# LONGITUDE raw and absolute statistics
print(f"\n{'=' * 110}")
print("LONGITUDE - RAW VALUES (all negative = Western US)")
print(f"{'=' * 110}")
long_raw = df['LONGITUDE']
print(f"  Count:  {long_raw.count():>12,.0f}")
print(f"  Min:    {long_raw.min():>16.8f}")
print(f"  Max:    {long_raw.max():>16.8f}")
print(f"  Mean:   {long_raw.mean():>16.8f}")
print(f"  Median: {long_raw.median():>16.8f}")
print(f"  Std:    {long_raw.std():>16.8f}")

print(f"\n{'=' * 110}")
print("LONGITUDE - ABSOLUTE VALUES")
print(f"{'=' * 110}")
long_abs = df['LONGITUDE'].abs()
print(f"  Count:  {long_abs.count():>12,.0f}")
print(f"  Min:    {long_abs.min():>16.8f}")
print(f"  Max:    {long_abs.max():>16.8f}")
print(f"  Mean:   {long_abs.mean():>16.8f}")
print(f"  Median: {long_abs.median():>16.8f}")
print(f"  Std:    {long_abs.std():>16.8f}")

print(f"\n  (Raw longitude range: {long_raw.min():.6f} to {long_raw.max():.6f})")
print(f"  All values are negative, indicating Western US locations")
print(f"{'=' * 110}")

# Check for any nulls
print(f"\n{'=' * 110}")
print("NULL VALUE COUNTS")
print(f"{'=' * 110}")
null_found = False
for col in cols:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        print(f"  {col}: {nulls} nulls")
        null_found = True
if not null_found:
    print("  No null values found in any of the selected columns.")
print(f"{'=' * 110}")

# Full describe output
print(f"\n{'=' * 110}")
print("FULL PANDAS DESCRIBE() OUTPUT")
print(f"{'=' * 110}")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', '{:.8f}'.format)
stats_df = df[cols].describe().transpose()
stats_df['unique'] = df[cols].nunique()
stats_df['count'] = stats_df['count'].astype(int)
print(stats_df[['count', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print(f"\n{'=' * 110}")
print("DONE")
print(f"{'=' * 110}")
