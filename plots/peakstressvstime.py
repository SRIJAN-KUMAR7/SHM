import numpy as np
import matplotlib.pyplot as plt

# Constants
sigma_0 = 5  # MPa
alpha = 0.4
beta = 2
p = 4
tau_0 = 2
L_ref = 1
Ea = 50000
R = 8.314
T_ref = 298
n = 1

crack_length = 2.0  # mm
temperature = 310   # K
t_values = np.linspace(0, 24, 100)
tau = tau_0 * (L_ref / crack_length)**n * np.exp((Ea / R) * (1 / temperature - 1 / T_ref))
bt = 1 - np.exp(-t_values / tau)
phi = (bt**p) / ((1 - bt)**p + bt**p)
sigma_t = sigma_0 * phi * (alpha + beta / crack_length)


plt.figure(figsize=(10, 5))
plt.plot(t_values, sigma_t, color='blue', label='Peak Stress σ(t)')
plt.title("Peak Stress vs Time")
plt.xlabel("Healing Time (hours)")
plt.ylabel("Peak Stress σ(t) [MPa]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
