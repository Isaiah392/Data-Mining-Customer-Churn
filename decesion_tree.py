
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt



# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Quick overview
print(df.shape)
print(df.head())
print(df.info())

######################################
##########DATA PRE-PROCESSING#########
#Cleaning up total charges coloumn because it has some spaces supposedly
#Dropping customer ID since it is irrelevant to the training process
#enchoding target label, and one hot encoding categorical features


# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split into X and y
X = df.drop('Churn', axis=1)
y = df['Churn']


###############################
#SPLITTING THE DATASET NEXT#### (80/20)like we usually do
###############################


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


###################
##TRAINING THE TREE (depth of 5 should probably do different depths, using gini)
###################
dt = DecisionTreeClassifier(
    criterion='entropy',  # or "gini"
    max_depth=5,
    random_state=42
)
dt.fit(X_train, y_train)



################
###EVALUATION
###############


y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#########################################
####VISUALIZATION AND FEATURE IMPORTANCE#
#########################################

plt.figure(figsize=(20,10))
tree.plot_tree(dt, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()


importances = pd.Series(dt.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()

importances = pd.Series(dt.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()