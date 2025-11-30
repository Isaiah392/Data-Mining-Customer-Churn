import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



# adjust path if needed
df = pd.read_csv("Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)


######CLEANING THE DATA########
# Drop ID column (not predictive)
df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric (coerce errors to NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows where TotalCharges could not be converted
df = df.dropna(subset=["TotalCharges"])

print("\n\n AFTER CLEANING TOTAL CHARGES\n")

print("After cleaning:", df.shape)
print(df.dtypes)


####### DEFINING LABEL AND FEATURES ######

print("Setting up label \n\n")

# Label: convert "Churn" from Yes/No to 1/0
y = df["Churn"].map({"No": 0, "Yes": 1})

# Features: everything except Churn
X = df.drop(columns=["Churn"])

print("\n\n")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sample y values:", y.value_counts())


########DIVIDING DATA SET INTO TRAIN AND TEST ########
print("\n\nDIVING DATA INTO TRAIN AND TEST 80/20\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 20% test
    random_state=42,     # reproducible split
    stratify=y           # keep churn ratio similar in train and test
)

print("Train size:", X_train.shape, y_train.shape)
print("Test size:", X_test.shape, y_test.shape)



print("\n\n More preprocessing. \n one hot encoding categorical features.\n")



##### DOING MORE PRE-PROCESSING #########
# Find categorical and numeric columns automatically
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_cols , "\n\n")
print("Numeric columns:", numeric_cols)

# Preprocessing: OneHotEncode categoricals, passthrough numerics
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

print("\n\n")



print("Building and training the random forest pipeline \n\n")

# Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,     # number of trees
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

# Full pipeline: preprocessing -> model
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf)
])

# Train (fit) the model
clf.fit(X_train, y_train)


print("FINISHED!!!\n\n")



#####Evauluating the model #########
from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))



#######FEATURE IMPORTANCE FROM NUMPY

# Get trained RandomForest from the pipeline
rf_model = clf.named_steps["model"]

import numpy as np

importances = rf_model.feature_importances_
print("Number of features after encoding:", len(importances))
