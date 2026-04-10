import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==============================
# TITLE
# ==============================
st.title("Customer Churn Prediction & Sales Dashboard")

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("customer.csv")

data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
data.ffill(inplace=True)

le = LabelEncoder()
for col in ['Location', 'Gender', 'Card Type']:
    data[col] = le.fit_transform(data[col])

# ==============================
# FEATURE & TARGET
# ==============================
X = data.drop("Exited", axis=1)
y = data["Exited"]

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# CACHE MODEL TRAINING (IMPORTANT)
# ==============================
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train)
    log_acc = log_model.score(X_test, y_test)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_acc = rf_model.score(X_test, y_test)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_acc = xgb_model.score(X_test, y_test)

    ann_model = Sequential()
    ann_model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    ann_model.add(Dense(8, activation='relu'))
    ann_model.add(Dense(1, activation='sigmoid'))

    ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ann_model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)

    _, ann_acc = ann_model.evaluate(X_test, y_test)

    return log_model, rf_model, xgb_model, ann_model, log_acc, rf_acc, xgb_acc, ann_acc

# CALL MODELS
log_model, rf_model, xgb_model, ann_model, log_acc, rf_acc, xgb_acc, ann_acc = train_models(
    X_train, X_test, y_train, y_test
)

# ==============================
# MODEL RESULTS
# ==============================
st.subheader("Model Comparison")
st.write("Logistic Regression:", log_acc)
st.write("Random Forest:", rf_acc)
st.write("XGBoost:", xgb_acc)
st.write("Neural Network:", ann_acc)

# ==============================
# VISUALS
# ==============================
st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Exited', data=data, ax=ax1)
st.pyplot(fig1)

st.subheader("Age Distribution")
fig2, ax2 = plt.subplots()
data['Age'].hist(ax=ax2)
st.pyplot(fig2)

st.subheader("Balance vs Churn")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Exited', y='Account Balance', data=data, ax=ax3)
st.pyplot(fig3)

# ==============================
# SEGMENTATION
# ==============================
st.subheader("Customer Segmentation")

cluster_data = data[['CreditScore', 'Age', 'Account Balance', 'EstimatedSalary']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(cluster_data)

fig4, ax4 = plt.subplots()
sns.scatterplot(x='Age', y='Account Balance', hue='Cluster', data=data, ax=ax4)
st.pyplot(fig4)

# ==============================
# SALES TREND
# ==============================
st.subheader("Sales Trend")

data['Month'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
data['Month'] = data['Month'].dt.to_period('M')

monthly_sales = data.groupby('Month')['EstimatedSalary'].sum()
st.line_chart(monthly_sales)

# ==============================
# INPUT
# ==============================
st.subheader("Predict Customer Churn")

credit = st.number_input("Credit Score", 0.0)
location = st.number_input("Location (0/1/2)", 0.0)
gender = st.number_input("Gender (0/1)", 0.0)
age = st.number_input("Age", 0.0)
tenure = st.number_input("Tenure", 0.0)
balance = st.number_input("Account Balance", 0.0)
products = st.number_input("Num Of Products", 0.0)
card = st.number_input("Has Credit Card (0/1)", 0.0)
active = st.number_input("Is Active Member (0/1)", 0.0)
salary = st.number_input("Estimated Salary", 0.0)
complain = st.number_input("Complain (0/1)", 0.0)
satisfaction = st.number_input("Satisfaction Score", 0.0)
cardtype = st.number_input("Card Type (0/1/2)", 0.0)
points = st.number_input("Points Earned", 0.0)

model_choice = st.selectbox("Select Model",
                           ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"])

# ==============================
# PREDICTION (FINAL FIXED)
# ==============================
if st.button("Predict", key="predict_button"):

    input_data = np.array([[credit, location, gender, age, tenure, balance,
                            products, card, active, salary, complain,
                            satisfaction, cardtype, points]])

    input_scaled = scaler.transform(input_data)

    try:
        if model_choice == "Logistic Regression":
            prediction = log_model.predict(input_scaled)[0]

        elif model_choice == "Random Forest":
            prediction = rf_model.predict(input_scaled)[0]

        elif model_choice == "XGBoost":
            prediction = xgb_model.predict(input_scaled)[0]

        else:
            prediction = (ann_model.predict(input_scaled) > 0.5).astype(int)[0][0]

        if prediction == 1:
            st.error("Customer will churn ❌")
        else:
            st.success("Customer will stay ✅")

    except Exception as e:
        st.error(f"Error: {e}")