import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


np.random.seed(42)


data_size = 1000
claim_amount = np.random.uniform(100, 5000, data_size)  # Random claim amounts
accident_severity = np.random.randint(1, 6, data_size)  # Severity levels 1 to 5
clmage = np.random.randint(18, 80, data_size)  # Age range 18 to 80
clmsex = np.random.randint(0, 2, data_size)  # 0 for Male, 1 for Female
clmins = np.random.randint(0, 2, data_size)  # 0 for No Insurance, 1 for Insurance
attorney_involved = np.random.randint(0, 2, data_size)  # 0 for Not Involved, 1 for Involved


df = pd.DataFrame({
    'Claim_Amount_Requested': claim_amount,
    'Accident_Severity': accident_severity,
    'CLMAGE': clmage,
    'CLMSEX': clmsex,
    'CLMINSUR': clmins,
    'Attorney_Involved': attorney_involved
})

# Split data into features (X) and target (y)
X = df[['Claim_Amount_Requested', 'Accident_Severity', 'CLMAGE', 'CLMSEX', 'CLMINSUR']]
y = df['Attorney_Involved']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as model.pkl and scaler.pkl")
