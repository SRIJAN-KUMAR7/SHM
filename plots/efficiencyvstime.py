import numpy as np
import matplotlib.pyplot as plt
eta_max = 0.95
tau_0 = 2
L_ref = 1
Ea = 50000
R = 8.314
T_ref = 298
n = 1
# Input values
crack_length = 2.0  # mm
temp = 310   # K
t_values = np.linspace(0, 24, 100)
tau = tau_0 * (L_ref / crack_length)**n * np.exp((Ea / R) * (1 / temp - 1 / T_ref))
eta_values = eta_max * (1 - np.exp(-t_values / tau))
plt.figure(figsize=(10, 5))
plt.plot(t_values, eta_values, color='green', label='Healing Efficiency η(t)')
plt.title("Healing Efficiency vs Time")
plt.xlabel("Healing Time (hours)")
plt.ylabel("Healing Efficiency η(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
