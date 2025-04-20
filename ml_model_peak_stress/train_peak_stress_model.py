import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os
np.random.seed(42)
num_samples = 500

crack_lengths = np.random.uniform(0.5, 5.0, num_samples)  
temp = np.random.uniform(290, 340, num_samples)   

#some values 
sigma_0 = 5        
tau_0 = 2
L_ref = 1
Ea = 50000         
R = 8.314
T_ref = 298
alpha = 0.4
beta = 2
n = 1
p = 4
t_heal = 6        

taus = tau_0 *(L_ref / crack_lengths)**n * np.exp((Ea/R)*(1/temp - 1/T_ref))
bt = (1 - np.exp(-t_heal / taus))
phi = (bt**p) / ((1 - bt)**p + bt**p)
peak_stress = sigma_0 * phi * (alpha + beta / crack_lengths)


df = pd.DataFrame({
    "crack_length": crack_lengths,
    "temperature": temp,
    "peak_stress": peak_stress
})

X = df[["crack_length", "temperature"]]
y = df["peak_stress"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


try:
    cl = float(input("\nEnter crack length (in mm): "))
    temp = float(input("Enter temperature (in Kelvin): "))

    user_input = pd.DataFrame([[cl, temp]], columns=["crack_length", "temperature"])
    predicted_stress = model.predict(user_input)

    print(f"\n Predicted Peak Stress for Crack = {cl} mm, Temp = {temp} K: {predicted_stress[0]} MPa")
except ValueError:
    print("Please enter valid numeric inputs for crack length and temperature.")
y_pred = model.predict(X_test)
print(" R² Score on Test Set:", r2_score(y_test, y_pred))
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/peak_stress_model.pkl")

try:
    cl = float(input("\nEnter crack length (in mm): "))
    temp = float(input("Enter temperature (in Kelvin): "))
    user_input = pd.DataFrame([[cl, temp]], columns=["crack_length", "temperature"])
    predicted_stress = model.predict(user_input)
    print(f"\n Predicted Peak Stress for Crack = {cl} mm, Temp = {temp} K: {predicted_stress[0]} MPa")
except ValueError:
    print("Please enter valid numeric inputs for crack length and temperature.")
