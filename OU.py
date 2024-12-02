import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def simulate_ou_process(theta, mu, sigma, x0, dt, n_steps, key):
    """
    Simulate an Ornstein-Uhlenbeck process using Euler-Maruyama method.

    Parameters:
        theta (float): Mean reversion rate.
        mu (float): Long-term mean.
        sigma (float): Volatility.
        x0 (float): Initial value of the process.
        dt (float): Time step size.
        n_steps (int): Number of steps to simulate.
        key (jax.random.PRNGKey): JAX random key.

    Returns:
        jnp.ndarray: Simulated process values.
    """
    # Generate random normal increments
    key, subkey = jax.random.split(key)
    dW = jax.random.normal(subkey, shape=(n_steps,)) * jnp.sqrt(dt)

    # Initialize the process
    x = jnp.zeros(n_steps + 1)
    x = x.at[0].set(x0)

    # Simulate the process
    for i in range(n_steps):
        x = x.at[i + 1].set(x[i] + theta * (mu - x[i]) * dt + sigma * dW[i])

    return x


# Parameters
theta = 0.5  # Mean reversion rate
mu = 0.0  # Long-term mean
sigma = 0.1  # Volatility
x0 = 1.0  # Initial value
dt = 0.01  # Time step
n_steps = 5000  # Number of steps
key = jax.random.PRNGKey(42)  # Random seed

# Simulate
ou_process = simulate_ou_process(theta, mu, sigma, x0, dt, n_steps, key)

# Time axis
time = jnp.linspace(0, n_steps * dt, n_steps + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, ou_process, label="Ornstein-Uhlenbeck Process", color="blue")
plt.title("Simulated Ornstein-Uhlenbeck Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.axhline(mu, color="red", linestyle="--", label="Long-term Mean ($\mu$)")
plt.legend()
plt.grid(True)
plt.show()
