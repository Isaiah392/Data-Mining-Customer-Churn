import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

# adjust path if needed
df = pd.read_csv("Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)
