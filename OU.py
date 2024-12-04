import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
theta = 0.5
mu = 0
sigma = 1
X0_mean = 10
X0_std = np.sqrt(10)
time_steps = [10, 40, 60, 80, 100, 150, 200, 1000]
num_samples = 1000
dt = 0.05

# Initialize X_0
X0 = np.random.normal(X0_mean, X0_std, num_samples)


# Function to simulate OU process
def simulate_ou_process(X0, theta, mu, sigma, dt, num_steps):
    X = np.zeros((num_steps, num_samples))
    X[0] = X0
    for t in range(1, num_steps):
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal(size=num_samples)
    return X


# Simulate the process
num_steps = max(time_steps) + 1
X = simulate_ou_process(X0, theta, mu, sigma, dt, num_steps)


# Function to calculate marginal distribution parameters
def marginal_distribution_params(mu, S, sigma, theta, t):
    alpha_t = np.exp(-theta * t)
    sigma_t = np.sqrt((1 - np.exp(-2 * theta * t)) / (2 * theta)) * sigma
    return alpha_t * mu, alpha_t ** 2 * S + sigma_t ** 2


# Plot the empirical distribution and marginal density function at specified time steps
plt.figure(figsize=(3*len(time_steps), 5))
for i, t in enumerate(time_steps):
    plt.subplot(1, len(time_steps), i + 1)
    plt.hist(X[t], bins=20, density=True, alpha=0.6, color='g', label='Empirical')

    # Calculate marginal distribution parameters
    mu_t, S_t = marginal_distribution_params(X0_mean, X0_std ** 2, sigma, theta, t)

    # Plot marginal density function
    x = np.linspace(min(X[t]), max(X[t]), 100)
    plt.plot(x, norm.pdf(x, mu_t, np.sqrt(S_t)), 'r-', lw=2, label='Marginal Density')

    plt.title(f'Distribution at t={t}')
    plt.xlabel('X_t')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()