import numpy as np
import matplotlib.pyplot as plt
eta_max = 0.95
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
t_heal = 6  # hours

crack_lengths = np.linspace(0.5, 5.0, 100)
temperatures = [290, 300, 310, 320, 330]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Healing Efficiency ---
for T in temperatures:
    tau = tau_0 * (L_ref / crack_lengths)**n * np.exp((Ea / R) * (1 / T - 1 / T_ref))
    eta = eta_max * (1 - np.exp(-t_heal / tau))
    axes[0].plot(crack_lengths, eta, label=f"T = {T}K")

axes[0].set_title("Healing Efficiency vs Crack Length")
axes[0].set_xlabel("Crack Length (mm)")
axes[0].set_ylabel("Efficiency η(t)")
axes[0].legend()
axes[0].grid(True)

# --- Plot 2: Peak Stress ---
for T in temperatures:
    tau = tau_0 * (L_ref / crack_lengths)**n*np.exp((Ea/R)* (1/ T - 1/ T_ref))
    bt = 1 - np.exp(-t_heal / tau)
    phi = (bt**p) / ((1 - bt)**p + bt**p)
    sigma = sigma_0 * phi * (alpha + beta / crack_lengths)
    axes[1].plot(crack_lengths, sigma, label=f"T = {T}K")

axes[1].set_title("Peak Stress vs Crack Length")
axes[1].set_xlabel("Crack Length (mm)")
axes[1].set_ylabel("Peak Stress σ(t) [MPa]")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
