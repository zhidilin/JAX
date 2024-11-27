import functools
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
from scipy.stats import multivariate_normal as scio_mvn
import scipy.stats as stats


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# Plotting libraries
import matplotlib.pyplot as plt
plt.style.use(['seaborn-paper'])


def get_data(N=30, sigma_obs=0.15, N_test=400):
    """
    Generate synthetic data
    """
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    Y += sigma_obs * np.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = jnp.linspace(-1.2, 1.2, N_test)

    return X[:, None], Y[:, None], X_test[:, None], None


logger.setLevel(logging.INFO)
X, y, Xtest, ytest = get_data()
print(X.shape, y.shape)

fig, ax = plt.subplots()
ax.scatter(X, y, c='red')
plt.show()


# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x, y):
    return jnp.sum((x-y)**2)

# RBF Kernel
@jax.jit
def rbf_kernel(params, x, y):
    return jnp.exp( - params['gamma'] * sqeuclidean_distance(x, y))

# ARD Kernel
@jax.jit
def ard_kernel(params, x, y):
    # divide by the length scale
    x = x / params['length_scale']
    y = y / params['length_scale']
    # return the ard kernel
    return params['var_f'] * jnp.exp( - sqeuclidean_distance(x, y) )

def zero_mean(x):
    return jnp.zeros(x.shape[0])

def gp_prior(params, mu_f, cov_f, x):
    return mu_f(x) , cov_f(params, x, x)

# Gram Matrix
def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)

params = {
    'gamma': 1.0,
    'var_f': 1.0,
    'likelihood_noise': 0.01,
}

cov_f = functools.partial(gram, rbf_kernel)
K_ = cov_f(params, Xtest, X)
print(K_.shape)

K_ = cov_f(params, X, Xtest)
print(K_.shape)


# define mean function
mu_f = zero_mean

# checks - 1 vector (D)
test_X = X[0, :].copy()
mu_x, cov_x = gp_prior(params, mu_f=mu_f, cov_f=cov_f, x=test_X)

print(mu_x.shape, cov_x.shape)
assert mu_x.shape[0] == test_X.shape[0]
assert jnp.ndim(mu_x) == 1
# Check output shapes, # of dimensions
assert cov_x.shape[0] == test_X.shape[0]
assert jnp.ndim(cov_x) == 2


# checks - 1 vector with batch size (NxD)
test_X = X.copy()
mu_x, cov_x = gp_prior(params, mu_f=mu_f, cov_f=cov_f, x=test_X)
assert mu_x.shape[0] == test_X.shape[0]
assert jnp.ndim(mu_x) == 1
# Check output shapes, # of dimensions
assert cov_x.shape[0] == test_X.shape[0]
assert jnp.ndim(cov_x) == 2


mu_x, cov_x = gp_prior(params, mu_f=mu_f, cov_f=cov_f , x=test_X)

# make it semi-positive definite with jitter
jitter = 1e-6
cov_x_ = cov_x + jitter * np.eye(cov_x.shape[0])

# draw random samples from distribution
n_functions = 10
key = jax.random.PRNGKey(0)
y_samples = scio_mvn.rvs(mean=mu_x, cov=cov_x_ , size=n_functions)

print(y_samples.shape)
for isample in y_samples:
    plt.plot(isample)
plt.show()


def gp_prior(params, mu_f, cov_f, x):
    return mu_f(x) , cov_f(params, x, x)


def cholesky_factorization(K, Y):

    # cho factor the cholesky
    logger.debug(f"ChoFactor: K{K.shape}")
    L = jax.scipy.linalg.cho_factor(K, lower=True)
    logger.debug(f"Output, L: {L[0].shape}, {L[1]}")

    # weights
    logger.debug(f"Input, ChoSolve(L, Y): {L[0].shape, Y.shape}")
    weights = jax.scipy.linalg.cho_solve(L, Y)
    logger.debug(f"Output, alpha: {weights.shape}")

    return L, weights

jitter = 1e-6

def posterior(params, prior_params, X, Y, X_new, likelihood_noise=False):
    logging.debug(f"Inputs, X: {X.shape}, Y: {Y.shape}, X*: {X_new.shape}")
    (mu_func, cov_func) = prior_params
    logging.debug("Loaded mean and cov functions")

    # ==========================
    # 1. GP PRIOR
    # ==========================
    logging.debug(f"Getting GP Priors...")

    mu_x, Kxx = gp_prior(params, mu_f=mu_func, cov_f=cov_func, x=X)
    logging.debug(f"Output, mu_x: {mu_x.shape}, Kxx: {Kxx.shape}")

    # check outputs
    assert mu_x.shape == (X.shape[0],), f"{mu_x.shape} =/= {(X.shape[0],)}"
    assert Kxx.shape == (
        X.shape[0],
        X.shape[0],
    ), f"{Kxx.shape} =/= {(X.shape[0],X.shape[0])}"

    # ===========================
    # 2. CHOLESKY FACTORIZATION
    # ===========================
    logging.debug(f"Solving Cholesky Factorization...")

    # 1 STEP
#     print(f"Problem: {Kxx.shape},{Y.shape}")
    (L, lower), alpha = cholesky_factorization(
        Kxx + (params["likelihood_noise"] + 1e-6) * jnp.eye(Kxx.shape[0]), Y
    )
    logging.debug(f"Output, L: {L.shape}, alpha: {alpha.shape}")
    assert L.shape == (
        X.shape[0],
        X.shape[0],
    ), f"L:{L.shape} =/= X..:{(X.shape[0],X.shape[0])}"
    assert alpha.shape == (X.shape[0], 1), f"alpha: {alpha.shape} =/= X: {X.shape[0], 1}"

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================
    logging.debug(f"Getting Projection Kernel...")
    logging.debug(f"Input, cov(x*, X): {X_new.shape},{X.shape}")

    # calculate transform kernel
    KxX = cov_func(params, X_new, X)

    logging.debug(f"Output, KxX: {KxX.shape}")


    assert KxX.shape == (
        X_new.shape[0],
        X.shape[0],
    ), f"{KxX.shape} =/= {(X_new.shape[0],X.shape[0])}"

    # Project data
    logging.debug(f"Getting Predictive Mean Distribution...")
    logging.debug(f"Input, mu(x*): {X_new.shape}, KxX @ alpha: {KxX.shape} @ {alpha.shape}")
    mu_y = jnp.dot(KxX, alpha)
    logging.debug(f"Output, mu_y: {mu_y.shape}")
    assert mu_y.shape == (X_new.shape[0],1)

    # =====================================
    # 5. PREDICTIVE COVARIANCE DISTRIBUTION
    # =====================================
    logging.debug(f"Getting Predictive Covariance matrix...")
    logging.debug(f"Input, L @ KxX.T: {L.shape} @ {KxX.T.shape}")

    #     print(f"K_xX: {KXx.T.shape}, L: {L.shape}")
    v = jax.scipy.linalg.cho_solve((L, True), KxX.T)

    logging.debug(f"Output, v: {v.shape}")
    assert v.shape == (
        X.shape[0],
        X_new.shape[0],
    ), f"v: {v.shape} =/= {(X_new.shape[0])}"

    logging.debug(f"Covariance matrix tests...cov(x*, x*)")
    logging.debug(f"Inputs, cov(x*, x*) - {X_new.shape},{X_new.shape}")
    Kxx = cov_func(params, X_new, X_new)

    logging.debug(f"Output, Kxx: {Kxx.shape}")
    assert Kxx.shape == (X_new.shape[0], X_new.shape[0])

    logging.debug(f"Calculating final covariance matrix...")
    logging.debug(f"Inputs, Kxx: {Kxx.shape}, v:{v.shape}")

    cov_y = Kxx - jnp.dot(KxX, v)
    logging.debug(f"Output: cov(x*, x*) - {cov_y.shape}")

    assert cov_y.shape == (X_new.shape[0], X_new.shape[0])

    if likelihood_noise is True:
        cov_y += params['likelihood_noise']

    # TODO: Bug here for vmap...

    # =====================================
    # 6. PREDICTIVE VARIANCE DISTRIBUTION
    # =====================================
    logging.debug(f"Getting Predictive Variance...")
    logging.debug(f"Input, L.T, I: {L.T.shape}, {KxX.T.shape}")

    Linv = jax.scipy.linalg.solve_triangular(L.T, jnp.eye(L.shape[0]))


    logging.debug(f"Output, Linv: {Linv.shape}, {Linv.min():.2f},{Linv.max():.2f}")

    logging.debug(f"Covariance matrix tests...cov(x*, x*)")
    logging.debug(f"Inputs, cov(x*, x*) - {X_new.shape},{X_new.shape}")
    var_y = jnp.diag(cov_func(params, X_new, X_new))
    logging.debug(f"Output, diag(Kxx): {var_y.shape}, {var_y.min():.2f},{var_y.max():.2f}")

    logging.debug(f"Inputs, Linv @ Linv.T - {Linv.shape},{Linv.T.shape}")
    Kinv =  jnp.dot(Linv, Linv.T)
    logging.debug(f"Output, Kinv: {Kinv.shape}, {Kinv.min():.2f},{Kinv.max():.2f}")

    logging.debug(f"Final Variance...")
    logging.debug(f"Inputs, KxX: {KxX.shape}, {Kinv.shape}, {KxX.shape}")
    var_y -= jnp.einsum("ij,ij->i", jnp.dot(KxX, Kinv), KxX) #jnp.dot(jnp.dot(KxX, Kinv), KxX.T)
    logging.debug(f"Output, var_y: {var_y.shape}, {var_y.min():.2f},{var_y.max():.2f}")
    #jnp.einsum("ij, jk, ki->i", KxX, jnp.dot(Linv, Linv.T), KxX.T)

    return mu_y, cov_y, jnp.diag(cov_y)

logger.setLevel(logging.DEBUG)
# MEAN FUNCTION
mu_f = zero_mean

# COVARIANCE FUNCTION
params = {
    'gamma': 1.0,
    'var_f': 1.0,
    'likelihood_noise': 0.01,
}
cov_f = functools.partial(gram, rbf_kernel)

# input vector
# x_plot = jnp.linspace(X.min(), X.max(), 100)[:, None]
test_X = Xtest[0, :]

prior_funcs = (mu_f, cov_f)

mu_y, cov_y, var_y = posterior(params, prior_funcs, X, y, X_new=test_X)

print(mu_y.shape,  cov_y.shape, var_y.shape)

mu_y, cov_y, var_y = posterior(params, prior_funcs, X, y, Xtest, True)
print(mu_y.shape,  cov_y.shape, var_y.shape)
plt.plot(var_y.squeeze())
plt.show()

uncertainty = 1.96 * jnp.sqrt(var_y.squeeze())

plt.fill_between(Xtest.squeeze(), mu_y.squeeze() + uncertainty, mu_y.squeeze() - uncertainty, alpha=0.1)
plt.plot(Xtest.squeeze(), mu_y.squeeze(), label='Mean')
plt.show()