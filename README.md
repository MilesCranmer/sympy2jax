# sympy2jax

Turn SymPy expressions into parametrized, differentiable, vectorizable, JAX functions.

All SymPy floats become trainable parameters. All SymPy symbols are inputs to the Module.

## Installation

```bash
pip install git+https://github.com/MilesCranmer/sympy2jax.git
```

## Example

```python
import sympy
from sympy import symbols
import jax
import jax.numpy as jnp
from jax import jacobian, random
from sympy2jax import sympy2jax
```

Let's create a functionin SymPy:
```python
x, y = symbols('x y')
cosx = 1.0 * sympy.cos(x) + 3.2 * y
```
Let's get the JAX version. We pass the equation, and
the symbols required.
```python
f, params = sympy2jax(cosx, [x, y])
```
The order you supply the symbols is the same order
you should supply the features when calling
the function `f` (shape `[nrows, nfeatures]`).
In this case, features=2 for x and y.
The `params` in this case will be
`jnp.array([1.0, 3.2])`. You pass these parameters
when calling the function, which will let you change them
and take gradients.

Let's generate some JAX data to pass:
```python
key = random.PRNGKey(0)
X = random.normal(key, (1000, 2))
```

We can call the function with:
```python
f(X, params)
```

We can take gradients like so:
```python
jac_f = jacobian(f, argnums=1)
jac_f(X, params)
```
