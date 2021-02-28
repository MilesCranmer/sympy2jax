import numpy as np
from jax import numpy as jnp
from jax import random
from jax import grad, jacobian
import sympy
from sympy2jax import sympy2jax

x, y, z = sympy.symbols('x y z')

def test_example():
    cosx = 1.0 * sympy.cos(x)
    key = random.PRNGKey(0)
    X = random.normal(key, (100, 1))
    true = 1.0 * jnp.cos(X[:, 0])
    f, params = sympy2jax(cosx, [x])
    assert jnp.all(jnp.isclose(f(X, params), true)).item()

def test_grad():
    cosx = 1.0 * sympy.cos(x)
    key = random.PRNGKey(0)
    X = random.normal(key, (100, 1))
    true_grad = - 1.0 * jnp.sin(X[:, 0])
    f, params = sympy2jax(cosx, [x])
    grad_prediction = grad(lambda x: f(x[:, None], params).sum())(X[:, 0])
    assert jnp.all(jnp.isclose(grad_prediction, true_grad)).item()


def test_multiple():
    cosxyz = 1.0 * sympy.cos(x) - 3.2 * sympy.Abs(y)**z
    key = random.PRNGKey(0)
    X = random.normal(key, (100, 3))
    true = 1.0 * jnp.cos(X[:, 0]) - 3.2 * jnp.abs(X[:, 1])**X[:, 2]
    f, params = sympy2jax(cosxyz, [x, y, z])
    assert jnp.all(jnp.isclose(f(X, params), true)).item()

print("Test 1")
test_example()
print("Test 2")
test_grad()
print("Test 3")
test_multiple()
