import matplotlib
import matplotlib.pyplot as plt


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

######THIS IS OUR FIRST ITERATION OF THE DECISION TREE
######It came out with a 79% accuracy with a depth of 5 (used gini erroring)
######Take a look at the decision_tree_with pruning for a potential
######different depth that better fitted the data




#### CHECK THE backend display settings make sure its using
#### tkinter GUI not (default) inline.
print("Backend:", matplotlib.get_backend())

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
    criterion='gini',  # or "entropy" doesnt really matter pretty small dataset I think
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

plt.figure(figsize=(40, 24))  # go big
tree.plot_tree(
    dt,
    filled=True,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    fontsize=8
)
plt.savefig("tree.svg", bbox_inches="tight")   
plt.show()

# --- new figure for bar chart ---
plt.figure(figsize=(10, 6))  # set a different size for the bar chart
importances = pd.Series(dt.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.savefig("top_ten.svg", bbox_inches="tight") 
plt.tight_layout()
plt.show()