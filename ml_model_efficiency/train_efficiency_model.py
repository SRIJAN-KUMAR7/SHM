import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
np.random.seed(42)
num_samples = 500
crack_lengths = np.random.uniform(0.5, 5.0, num_samples)  
temperatures = np.random.uniform(290, 340, num_samples)   

eta_max = 0.95
tau_0 = 2
L_ref = 1
Ea = 50000 
R = 8.314
T_ref = 298
n = 1
t_heal = 6 

taus = tau_0 * (L_ref / crack_lengths)**n * np.exp((Ea / R) * (1 / temperatures - 1 / T_ref))
efficiencies = eta_max * (1 - np.exp(-t_heal / taus))

df = pd.DataFrame({
    "crack_length": crack_lengths,
    "temperature": temperatures,
    "healing_efficiency": efficiencies
})


X = df[["crack_length", "temperature"]]
y = df["healing_efficiency"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



import pandas as pd

try:
    crack_length = float(input("Enter crack length (in mm): "))
    temperature = float(input("Enter temperature (in Kelvin): "))

 
    new_input = pd.DataFrame([[crack_length, temperature]], columns=["crack_length", "temperature"])

    # Predict
    predicted_efficiency = model.predict(new_input)

    print(f"\n Predicted Healing Efficiency for Crack = {crack_length} mm, Temp = {temperature} K: {predicted_efficiency[0]}")

except ValueError:
    print("NO,Please enter valid numerical values.")

output_df = new_input.copy()
output_df["predicted_efficiency"] = predicted_efficiency
output_df.to_csv("last_prediction.csv", index=False)
print(" Prediction saved to 'last_prediction.csv'")

y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))


import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/efficiency_model.pkl")