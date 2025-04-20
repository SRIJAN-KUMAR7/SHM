import numpy as np
import matplotlib.pyplot as plt

# Constants
sigma_0 = 7   
alpha = 0.4
beta = 2
p = 4
tau_0 = 2
L_ref = 1
Ea = 50000        # J/mol
R = 8.314
T_ref = 298
n = 1
temperature = 310  # Kelvin
t_heal = 6         # Healing time (hours)

crack_lengths = np.linspace(0.5, 5.0, 100)
taus = tau_0 * (L_ref / crack_lengths)**n * np.exp((Ea / R) * (1 / temperature - 1 / T_ref))
bt = 1 - np.exp(-t_heal / taus)
phi = (bt**p) / ((1 - bt)**p + bt**p)
peak_stress = sigma_0*phi*(alpha + beta / crack_lengths)
plt.figure(figsize=(10, 5))
plt.plot(crack_lengths, peak_stress, color='red', label='Peak Stress σ vs Crack Length')
plt.title("Peak Stress vs Crack Length at 6 hours (T = 310K)")
plt.xlabel("Crack Length (mm)")
plt.ylabel("Peak Stress σ(t) [MPa]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
