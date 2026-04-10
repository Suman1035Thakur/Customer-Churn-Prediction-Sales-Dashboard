# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("customer.csv")

print("Dataset Loaded Successfully ✅")
print(data.head())

# ==============================
# DATA PREPROCESSING
# ==============================

# Drop unnecessary columns
data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Handle missing values (updated method)
data.ffill(inplace=True)

# Encode categorical columns
le = LabelEncoder()

categorical_cols = ['Location', 'Gender', 'Card Type']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

print("\nData Preprocessing Completed ✅")

# ==============================
# FEATURE & TARGET
# ==============================
X = data.drop("Exited", axis=1)
y = data["Exited"]

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL TRAINING
# ==============================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\nModel Trained Successfully ✅")
print("Model Accuracy:", accuracy)

# ==============================
# DASHBOARD VISUALIZATION
# ==============================

# 1. Churn Distribution
plt.figure()
sns.countplot(x='Exited', data=data)
plt.title("Churn Distribution (0 = No, 1 = Yes)")
plt.show()

# 2. Age Distribution
plt.figure()
data['Age'].hist()
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 3. Balance vs Churn
plt.figure()
sns.boxplot(x='Exited', y='Account Balance', data=data)
plt.title("Account Balance vs Churn")
plt.show()

# 4. Active Members vs Churn
plt.figure()
sns.countplot(x='IsActiveMember', hue='Exited', data=data)
plt.title("Active Members vs Churn")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()